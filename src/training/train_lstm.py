import torch
import torch.nn as nn
import numpy as np
import os
import yaml
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.models.lstm_model import build_lstm


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    processed_dir = config["data"]["processed_dir"]
    X = np.load(os.path.join(processed_dir, "X.npy"))
    y = np.load(os.path.join(processed_dir, "y.npy"))
    return X, y


def split_data(X, y, val_split: float, test_split: float):
    n           = len(X)
    test_start  = int(n * (1 - test_split))
    val_start   = int(n * (1 - test_split - val_split))

    X_train, y_train = X[:val_start],          y[:val_start]
    X_val,   y_val   = X[val_start:test_start], y[val_start:test_start]
    X_test,  y_test  = X[test_start:],          y[test_start:]

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_loader(X, y, batch_size: int, shuffle: bool):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )


def train(config: dict):
    epochs     = config["training"]["epochs"]
    lr         = config["training"]["learning_rate"]
    batch_size = config["training"]["batch_size"]
    patience   = config["training"]["early_stopping_patience"]
    val_split  = config["training"]["val_split"]
    test_split = config["training"]["test_split"]
    save_path  = config["training"]["model_save_path"]

    os.makedirs(save_path, exist_ok=True)

    # --- Load & split ---
    print("\nLoading data...")
    X, y = load_data(config)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, val_split, test_split
    )

    # --- Save test set ---
    # Note: same test set as transformer — fair comparison
    processed_dir = config["data"]["processed_dir"]
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)

    # --- Dataloaders ---
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   batch_size, shuffle=False)

    # --- Model ---
    model     = build_lstm(config)
    loss_fn   = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=3, factor=0.5, verbose=True
    )

    best_val_loss    = float("inf")
    patience_counter = 0

    print(f"\nTraining LSTM for up to {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):

        # -- Train --
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimiser.zero_grad()
            preds = model(X_batch)
            loss  = loss_fn(preds, y_batch)
            loss.backward()
            # Gradient clipping — same as transformer for fair comparison
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item()

        # -- Validate --
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds    = model(X_batch)
                val_loss += loss_fn(preds, y_batch).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        scheduler.step(avg_val)

        print(f"  Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {avg_train:.6f} | "
              f"Val Loss:   {avg_val:.6f}")

        # -- Early stopping --
        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(save_path, "best_lstm.pt")
            )
            print(f"             ✓ Best LSTM saved (val loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    print(f"\n✓ LSTM training complete. Best val loss: {best_val_loss:.6f}")
    print(f"  Model saved → models/best_lstm.pt")


if __name__ == "__main__":
    config = load_config()
    train(config)