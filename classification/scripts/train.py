import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import wandb
from torchvision.models import ResNet50_Weights

from classification.models.resnet import ResNet50MC
from classification.data_loaders.aptos_data_loader import get_aptos_loaders
from classification.utils.metrics import accuracy

def train(model, train_loader, val_loader, config, device):
    """
    Train a model given dataloaders and config.
    """

    model = model.to(device)

    # --- Loss function ---
    criterion = nn.CrossEntropyLoss()

    # --- Learning rates ---
    base_lr = float(config["lr"])
    warmup_lr = float(config["warmup_lr"])

    # --- Optimizer (only fc at start) ---
    optimizer = optim.AdamW(model.model.fc.parameters(), lr=warmup_lr, weight_decay=1e-4)

    # --- LR scheduler (Reduce on Plateau) ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3    
    )

    # --- Early stopping setup ---
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = config["early_stop_patience"]

    # --- CSV logging ---
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    log_csv_path = os.path.join(output_dir, "train_log.csv")
    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    # --- Training loop ---
    for epoch in range(config["epochs"]):
        # Warmup phase
        if epoch == config["warmup_epochs"]:
            # Unfreeze all layers
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

        # Training
        model.train()
        running_loss, running_acc = 0.0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_acc += accuracy(outputs, labels) * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_acc += accuracy(outputs, labels) * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Train loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # CSV logging
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, epoch_loss, epoch_acc, val_loss, val_acc])

        # --- Wandb logging ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        # LR scheduler update
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save the best model
    save_model(best_model_state, output_dir, filename="model.pth")

def save_model(model_state, output_dir, filename):
    """
    Save the model state dict to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, filename)
    torch.save(model_state, model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Data loaders ---
    if config["dataset"].lower() == "aptos2019":
        train_loader, val_loader, _, num_classes = get_aptos_loaders(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"]
        )
    else:
        raise ValueError(f"Dataset {config['dataset']} not supported.")

    # --- Model ---
    model = ResNet50MC(
        num_classes=num_classes,
        weights=ResNet50_Weights.DEFAULT,
        dropout_p=config["dropout"]
    )

    wandb.init(
        project="resnet50-" + config["dataset"].lower(),
        config=config
    )
    config = wandb.config

    # --- Start training ---
    train(model, train_loader, val_loader, config, device)
