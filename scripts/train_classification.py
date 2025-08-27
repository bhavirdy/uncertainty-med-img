import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import time
from torchvision.models import ResNet50_Weights

from models.resnet import get_resnet50
from utils.aptos_data_loader import get_aptos_loaders
from utils.metrics import accuracy

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize W&B
    wandb.init(project="resnet50-finetuned", config=vars(args))
    config = wandb.config

    # --- Get data loaders based on dataset ---
    if args.dataset.lower() == "aptos2019":
        train_loader, val_loader, _, num_classes = get_aptos_loaders()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # --- Model + optimizer ---
    model, param_groups = get_resnet50(
        num_classes=num_classes,
        weights=ResNet50_Weights.DEFAULT,
        finetune=True,
        lr=config.lr,
        layer4_lr_factor=0.1
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(param_groups)

    # --- Training loop ---
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(config.epochs):
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
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

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
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"Train loss: {epoch_loss:.4f} acc: {epoch_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        # W&B logging
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

    # Save model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"./outputs/models/model_ResNet50_{args.dataset}_{timestamp}_valacc{val_acc:.4f}.pth"
    torch.save(model.state_dict(), model_filename)
    wandb.save("model.pth")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetuned ResNet50 classifier")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name, e.g. aptos2019')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data folder')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test data folder')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_split', type=float, default=0.1)
    args = parser.parse_args()

    train(args)
