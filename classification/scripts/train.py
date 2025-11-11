import argparse
import os
import csv
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy

from classification.models.resnet import ResNet50Deterministic, ResNet50MCDO, ResNet50EDL
from classification.data_loaders.aptos_data_loader import get_aptos_loaders
from classification.data_loaders.isic2018_data_loader import get_isic2018_loaders
from classification.utils.edl_loss import evidential_loss

def train(model, train_loader, val_loader, device, args):
    # --- Initialize wandb ---
    wandb.init(
        project=f"resnet50-{args.dataset.lower()}-{args.method}",
        config=vars(args)
    )

    model = model.to(device)

    base_lr = args.lr
    warmup_lr = args.warmup_lr

    optimizer = optim.AdamW(model.model.fc.parameters(), lr=warmup_lr, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_acc_metric = Accuracy(task="multiclass", num_classes=args.num_classes).to(device)
    val_acc_metric = Accuracy(task="multiclass", num_classes=args.num_classes).to(device)

    best_val_loss = float('inf')
    early_stop_counter = 0

    os.makedirs(args.output_dir, exist_ok=True)
    log_csv_path = os.path.join(args.output_dir, "train_log.csv")
    with open(log_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    for epoch in range(args.epochs):
        # --- Post-warmup: unfreeze backbone ---
        if epoch == args.warmup_epochs:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)

        # --- Training ---
        model.train()
        train_loss = 0.0
        train_acc_metric.reset()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)

            if args.method == 'edl':
                loss = evidential_loss(
                    alpha=outputs,
                    target=labels,
                    num_classes=args.num_classes,
                    epoch=epoch,
                    zeta=args.zeta
                )
            else:
                loss = nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            optimizer.step()

            if args.method == 'edl':
                probs = outputs / outputs.sum(dim=1, keepdim=True)
                preds = torch.argmax(probs, dim=1)
            else:
                preds = torch.argmax(outputs, dim=1)

            train_acc_metric.update(preds, labels)
            train_loss += loss.item() * imgs.size(0)

        avg_train_acc = train_acc_metric.compute().item()
        avg_train_loss = train_loss / len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_acc_metric.reset()

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                
                outputs = model(imgs)

                if args.method == 'edl':
                    loss = evidential_loss(
                        alpha=outputs,
                        target=labels,
                        num_classes=args.num_classes,
                        epoch=epoch,
                        zeta=args.zeta
                    )
                else:
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                if args.method == 'edl':
                    probs = outputs / outputs.sum(dim=1, keepdim=True)
                    preds = torch.argmax(probs, dim=1)
                else:
                    preds = torch.argmax(outputs, dim=1)

                val_acc_metric.update(preds, labels)
                val_loss += loss.item() * imgs.size(0)

        avg_val_acc = val_acc_metric.compute().item()
        avg_val_loss = val_loss / len(val_loader.dataset)

        # scheduler.step(avg_val_loss)
        scheduler.step()

        print(f"Epoch: {epoch+1}/{args.epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        # --- Logging ---
        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc
        })

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            save_model(model.state_dict(), args.output_dir, filename="model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    wandb.finish()

def save_model(model_state, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, filename)
    torch.save(model_state, model_path)
    print(f"Saved model to {model_path}")

def main():
    parser = argparse.ArgumentParser()

    # --- Arguments ---
    parser.add_argument('--dataset', type=str, required=True, choices=['aptos2019', 'isic2018'])
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    
    parser.add_argument("--method", type=str, required=True, choices=["deterministic", "mcdo", "edl"])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--zeta', type=float, default=0.1)
    
    args = parser.parse_args()

    # --- Device setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data loaders ---
    if args.dataset.lower() == "aptos2019":
        train_loader, val_loader, _ = get_aptos_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset.lower() == "isic2018":
        train_loader, val_loader, _ = get_isic2018_loaders(batch_size=args.batch_size, num_workers=args.num_workers)

    # --- Model ---
    if args.method == "deterministic":
        model = ResNet50Deterministic(num_classes=args.num_classes)
    elif args.method == "mcdo":
        model = ResNet50MCDO(num_classes=args.num_classes, dropout_p=args.dropout)
    elif args.method == "edl":
        model = ResNet50EDL(num_classes=args.num_classes)
        
    # --- Train ---
    train(model, train_loader, val_loader, device, args)

if __name__ == "__main__":
    main()
