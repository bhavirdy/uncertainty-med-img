import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import wandb
from torchvision.models import ResNet50_Weights

from classification.models.resnet import ResNet50MC
from classification.data_loaders.aptos_data_loader import get_aptos_loaders
from classification.data_loaders.isic2018_data_loader import get_isic2018_loaders
from classification.utils.metrics import accuracy

def train(model, train_loader, val_loader, device, args):
    """Train the model using specified arguments."""

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    base_lr = args.lr
    warmup_lr = args.warmup_lr

    optimizer = optim.AdamW(model.model.fc.parameters(), lr=warmup_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None

    os.makedirs(args.output_dir, exist_ok=True)
    log_csv_path = os.path.join(args.output_dir, "train_log.csv")
    with open(log_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    for epoch in range(args.epochs):
        # --- Warmup ---
        if epoch == args.warmup_epochs:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

        # --- Training ---
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

        # --- Validation ---
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

        print(f"Epoch {epoch+1}/{args.epochs} "
              f"Train: loss={epoch_loss:.4f} acc={epoch_acc:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f}")

        # CSV + WandB logging
        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, epoch_loss, epoch_acc, val_loss, val_acc])

        wandb.log({
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        scheduler.step(val_loss)

        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Save best model
    if best_model_state:
        save_model(best_model_state, args.output_dir, filename="model.pth")

def save_model(model_state, output_dir, filename):
    """Save model state dict to specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, filename)
    torch.save(model_state, model_path)
    print(f"Saved model to {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 for classification")

    # --- Arguments ---
    parser.add_argument('--dataset', type=str, required=True, choices=['aptos2019', 'isic2018'], help='Dataset name')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save logs and model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-3, help='Warmup learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--early_stop_patience', type=int, default=7, help='Early stopping patience')
    args = parser.parse_args()

    # --- Device setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data loaders ---
    if args.dataset.lower() == "aptos2019":
        train_loader, val_loader, _, num_classes = get_aptos_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset.lower() == "isic2018":
        train_loader, val_loader, _, num_classes = get_isic2018_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # --- Model ---
    model = ResNet50MC(
        num_classes=num_classes,
        weights=ResNet50_Weights.DEFAULT,
        dropout_p=args.dropout
    )

    # --- WandB init ---
    wandb.init(
        project="resnet50-" + args.dataset.lower(),
        config=vars(args)
    )

    # --- Train ---
    train(model, train_loader, val_loader, args, device)

if __name__ == "__main__":
    main()
