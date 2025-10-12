import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna
from sklearn.model_selection import train_test_split
import time

# ============================
# Device Setup (MPS)
# ============================
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"[INFO] Using device: {device}")

# ============================
# SMAPE Metric
# ============================
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    return np.mean(np.where(denominator == 0, 0, diff / denominator)) * 100

# ============================
# Activation Function Mapper
# ============================
def get_activation(name):
    return {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'tanh': nn.Tanh()
    }.get(name, nn.ReLU())

# ============================
# Flexible MLP Model
# ============================
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, activation):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ============================
# Load Dataset
# ============================
df = pd.read_csv('/Users/abhinavgupta/Desktop/ml/final_training_data.csv')  # Update path
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================
# Optuna Objective Function
# ============================
def objective(trial):
    # Hyperparameters
    hidden_dim = trial.suggest_int('hidden_dim', 64, 1024)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    dropout = trial.suggest_float('dropout', 0.0, 0.7)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'tanh'])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 10, 50)

    # Model
    model = MLPRegressor(input_dim=X_train.shape[1], hidden_dim=hidden_dim,
                         num_layers=num_layers, dropout=dropout, activation=activation)
    model.to(device)

    # Optimizer & Loss
    criterion = nn.MSELoss()
    optimizer = {
        'Adam': optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
        'SGD': optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay),
        'RMSprop': optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    }[optimizer_name]

    # DataLoader
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
    valid_dataset = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().flatten()
            preds.append(pred)
            targets.append(yb.numpy().flatten())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    score = smape(targets, preds)
    return score

# ============================
# Run Optuna
# ============================
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("Best trial:", study.best_trial.number)
print("Best SMAPE:", study.best_trial.value)
print("Best params:", study.best_trial.params)

# ============================
# Retrain and Save Best Model
# ============================
best_params = study.best_trial.params

model = MLPRegressor(
    input_dim=X_train.shape[1],
    hidden_dim=best_params['hidden_dim'],
    num_layers=best_params['num_layers'],
    dropout=best_params['dropout'],
    activation=best_params['activation']
)
model.to(device)

criterion = nn.MSELoss()
optimizer = {
    'Adam': optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay']),
    'SGD': optim.SGD(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay']),
    'RMSprop': optim.RMSprop(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
}[best_params['optimizer']]

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

for epoch in range(best_params['epochs']):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

# Save the trained model
save_path = "best_ann_model.pth"
torch.save(model.state_dict(), save_path)
print(f"Best model saved to {save_path}")
