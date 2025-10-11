# ============================
# Imports
# ============================
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import joblib

import torch
from transformers import CLIPProcessor, CLIPModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
import optuna

# ============================
# Parameters
# ============================
DATA_CSV = "dataset.csv"  # CSV with columns: image_path, description, price
IMAGE_COLUMN = "image_path"
TEXT_COLUMN = "description"
PRICE_COLUMN = "price"

DEVICE = "cpu"  # CPU-friendly
BATCH_SIZE = 16

SAVE_EMBEDDINGS = True
SAVE_MODEL = True
SAVE_FINAL_DATA = True

EMBEDDINGS_FILE = "features.npy"
PRICES_FILE = "prices.npy"
FINAL_DATA_NPY = "final_training_data.npy"
FINAL_DATA_CSV = "final_training_data.csv"
MODEL_FILE = "lgbm_price_model.pkl"

N_TRIALS = 50  # Optuna trials (increase if CPU allows)

# ============================
# Load pre-trained CLIP model
# ============================
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ============================
# Load dataset
# ============================
df = pd.read_csv(DATA_CSV)
print(f"Total samples: {len(df)}")

# ============================
# Function to compute SMAPE
# ============================
def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    """
    return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

# ============================
# Function to get embeddings
# ============================
def get_clip_embeddings(image_paths, texts, batch_size=16):
    image_embeddings = []
    text_embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Generating embeddings"):
        batch_images = []
        batch_texts = texts[i:i+batch_size]

        # Load images
        for img_path in image_paths[i:i+batch_size]:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                img = Image.new('RGB', (224, 224), color=(0,0,0))
            batch_images.append(img)

        # Process batch
        inputs = clip_processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            img_emb = clip_model.get_image_features(inputs['pixel_values'])
            txt_emb = clip_model.get_text_features(inputs['input_ids'], inputs['attention_mask'])

        image_embeddings.append(img_emb.cpu().numpy())
        text_embeddings.append(txt_emb.cpu().numpy())

    image_embeddings = np.vstack(image_embeddings)
    text_embeddings = np.vstack(text_embeddings)

    return image_embeddings, text_embeddings

# ============================
# Generate embeddings
# ============================
print("Generating embeddings for all samples...")
image_embs, text_embs = get_clip_embeddings(
    df[IMAGE_COLUMN].tolist(),
    df[TEXT_COLUMN].tolist(),
    batch_size=BATCH_SIZE
)

# ============================
# Concatenate embeddings
# ============================
features = np.concatenate([image_embs, text_embs], axis=1)
target = df[PRICE_COLUMN].values
print(f"Features shape: {features.shape}, Target shape: {target.shape}")

# ============================
# Save embeddings and final data
# ============================
if SAVE_EMBEDDINGS:
    np.save(EMBEDDINGS_FILE, features)
    np.save(PRICES_FILE, target)
    print(f"Saved embeddings to {EMBEDDINGS_FILE} and prices to {PRICES_FILE}")

if SAVE_FINAL_DATA:
    final_data = np.concatenate([features, target.reshape(-1, 1)], axis=1)
    np.save(FINAL_DATA_NPY, final_data)
    print(f"Saved final training data as {FINAL_DATA_NPY}")
    columns = [f"feat_{i}" for i in range(features.shape[1])] + ["price"]
    pd.DataFrame(final_data, columns=columns).to_csv(FINAL_DATA_CSV, index=False)
    print(f"Saved final training data as {FINAL_DATA_CSV}")

# ============================
# Split train/validation for Optuna
# ============================
X_train, X_valid, y_train, y_valid = train_test_split(features, target, test_size=0.2, random_state=42)

# ============================
# Optuna objective function
# ============================
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }
    model = LGBMRegressor(**params, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return smape(y_valid, preds)

# ============================
# Run Optuna study
# ============================
print("Starting Optuna hyperparameter tuning...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=N_TRIALS)

print("Best hyperparameters:", study.best_params)

# ============================
# Train final model on full data
# ============================
best_params = study.best_params
final_model = LGBMRegressor(**best_params, n_jobs=-1)
final_model.fit(features, target)

# ============================
# Evaluate on holdout test set if available
# ============================
# Optional: If you have a separate test dataset, you can load it and compute SMAPE
# Example:
# X_test = ... (concatenated embeddings)
# y_test = ... (true prices)
# y_pred = final_model.predict(X_test)
# print("Test SMAPE:", smape(y_test, y_pred))

# ============================
# Save trained model
# ============================
if SAVE_MODEL:
    joblib.dump(final_model, MODEL_FILE)
    print(f"Saved trained LightGBM model to {MODEL_FILE}")

print("Pipeline completed successfully!")
