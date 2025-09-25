import argparse
import os
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import csv
from torchvision.models import ResNet50_Weights

from classification.models.resnet import ResNet50MC
from classification.data_loaders.aptos_data_loader import get_aptos_loaders
from classification.utils.metrics import accuracy

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(config):
    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Seed ---
    set_seed(config['seed'])

    # --- Initialize W&B ---
    wandb.init(project="resnet50-" + config['dataset'].lower(), config=config)
    cfg = wandb.config

    # --- Data loaders ---
    if cfg.dataset.lower() == "aptos2019":
        train_loader, val_loader, _, num_classes = get_aptos_loaders(
            batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )
    else:
        raise ValueError(f"Dataset {cfg.dataset} not supported.")

    # --- Model ---
    model = ResNet50MC(
        num_classes=num_classes,
        weights=ResNet50_Weights.DEFAULT,
        dropout_p=cfg.dropout
    )
    model = model.to(device)

    # --- Loss ---
    if cfg.loss.lower() == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss function not supported.")

    # --- Base learning rate ---
    lr_base = float(cfg.lr)

    # --- Define discriminative LRs ---
    layer_lrs = {
        "fc": lr_base * 100,     # classification head
        "layer4": lr_base * 10,
        "layer3": lr_base * 3,
        "layer2": lr_base,
        "layer1": lr_base * 0.3,
        "conv1": lr_base * 0.1,
    }

    # --- Optimizer (AdamW with weight decay) ---
    param_groups = []
    for layer_name, lr in layer_lrs.items():
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            param_groups.append({"params": layer.parameters(), "lr": lr})

    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    # --- Early stopping ---
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = cfg.early_stop_patience

    # --- Prepare CSV logging ---
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_csv_path = os.path.join(output_dir, "train_log.csv")
    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    # --- Unfreeze schedule ---
    unfreeze_schedule = cfg.get('unfreeze_schedule', {})

    # --- Training loop ---
    for epoch in range(cfg.epochs):
        # Gradual unfreezing
        if epoch in unfreeze_schedule:
            for layer_name in unfreeze_schedule[epoch]:
                if hasattr(model, layer_name):
                    for param in getattr(model, layer_name).parameters():
                        param.requires_grad = True
            print(f"Unfroze {unfreeze_schedule[epoch]} at epoch {epoch+1}")

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

        print(f"Epoch {epoch+1}/{cfg.epochs} - "
              f"Train loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # W&B logging
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # CSV logging
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, epoch_loss, epoch_acc, val_loss, val_acc])

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

    # Save best model
    model_filename = os.path.join(output_dir, "model.pth")
    torch.save(best_model_state, model_filename)
    wandb.save(model_filename)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config)
