# ============================
# Imports
# ============================
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import joblib
import requests
from io import BytesIO

import torch
from transformers import CLIPProcessor, CLIPModel

from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import optuna

# ============================
# Parameters
# ============================
DATA_CSV = "/Users/abhinavgupta/Desktop/ml/train.xlsx"  # columns: image_link, catalog_content, price
IMAGE_COLUMN = "image_link"
TEXT_COLUMN = "catalog_content"
PRICE_COLUMN = "price"

# Device selection: MPS > CUDA > CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")
BATCH_SIZE = 32  # Increase for faster processing if MPS memory allows

IMAGE_CACHE_DIR = "cached_images"

SAVE_EMBEDDINGS = True
SAVE_MODEL = True
SAVE_FINAL_DATA = True

EMBEDDINGS_FILE = "features.npy"
PRICES_FILE = "prices.npy"
FINAL_DATA_NPY = "final_training_data.npy"
FINAL_DATA_CSV = "final_training_data.csv"
MODEL_FILE = "lgbm_price_model.pkl"

N_TRIALS = 50  # Optuna trials
N_SPLITS = 5   # K-Fold CV splits

os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

# ============================
# Load CLIP model
# ============================
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ============================
# Load dataset
# ============================
df = pd.read_excel(DATA_CSV)
print(f"Total samples: {len(df)}")

# ============================
# SMAPE function
# ============================
def smape(y_true, y_pred):
    return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

# ============================
# Download image function
# ============================
def download_image(url, cache_dir=IMAGE_CACHE_DIR):
    filename = os.path.join(cache_dir, url.split("/")[-1])
    if not os.path.exists(filename):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(filename)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            img = Image.new('RGB', (224, 224), color=(0, 0, 0))
            img.save(filename)
    else:
        img = Image.open(filename).convert("RGB")
    return img

# ============================
# Generate CLIP embeddings with mixed precision
# ============================
def get_clip_embeddings(image_urls, texts, batch_size=32):
    image_embeddings = []
    text_embeddings = []

    for i in tqdm(range(0, len(image_urls), batch_size), desc="Generating embeddings"):
        batch_images = []
        batch_texts = texts[i:i+batch_size]

        for url in image_urls[i:i+batch_size]:
            img = download_image(url)
            batch_images.append(img)

        # Mixed precision for MPS
        with torch.autocast(device_type=DEVICE):
            inputs = clip_processor(
                text=batch_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(DEVICE)

            with torch.no_grad():
                img_emb = clip_model.get_image_features(inputs['pixel_values'])
                txt_emb = clip_model.get_text_features(inputs['input_ids'], inputs['attention_mask'])

        image_embeddings.append(img_emb.cpu().numpy())
        text_embeddings.append(txt_emb.cpu().numpy())

        # Optional: save intermediate embeddings every 500 batches
        if i % (500 * batch_size) == 0 and SAVE_EMBEDDINGS:
            np.save(EMBEDDINGS_FILE, np.vstack(image_embeddings))
            print(f"Saved intermediate embeddings at batch {i}")

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

# Save embeddings and final data
if SAVE_EMBEDDINGS:
    np.save(EMBEDDINGS_FILE, features)
    np.save(PRICES_FILE, target)
    print(f"Saved embeddings to {EMBEDDINGS_FILE} and prices to {PRICES_FILE}")

if SAVE_FINAL_DATA:
    final_data = np.concatenate([features, target.reshape(-1, 1)], axis=1)
    np.save(FINAL_DATA_NPY, final_data)
    columns = [f"feat_{i}" for i in range(features.shape[1])] + ["price"]
    pd.DataFrame(final_data, columns=columns).to_csv(FINAL_DATA_CSV, index=False)
    print(f"Saved final training data as {FINAL_DATA_CSV}")

# ============================
# Log-transform target
# ============================
target_log = np.log1p(target)

# ============================
# Optuna objective with K-Fold CV
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

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    smape_scores = []

    for train_idx, val_idx in kf.split(features):
        X_tr, X_val = features[train_idx], features[val_idx]
        y_tr, y_val = target_log[train_idx], target_log[val_idx]

        model = LGBMRegressor(**params, n_jobs=-1)
        model.fit(X_tr, y_tr)
        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        y_val_orig = np.expm1(y_val)
        smape_scores.append(smape(y_val_orig, y_pred))

    return np.mean(smape_scores)

# ============================
# Run Optuna study
# ============================
print("Starting Optuna hyperparameter tuning with K-Fold CV...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=N_TRIALS)

print("Best hyperparameters:", study.best_params)

# ============================
# Train final model on full data with log-transform
# ============================
best_params = study.best_params
final_model = LGBMRegressor(**best_params, n_jobs=-1)
final_model.fit(features, target_log)

# ============================
# Save trained model
# ============================
if SAVE_MODEL:
    joblib.dump(final_model, MODEL_FILE)
    print(f"Saved trained LightGBM model to {MODEL_FILE}")

print("Pipeline completed successfully! Optimized for MPS and mixed precision.")
