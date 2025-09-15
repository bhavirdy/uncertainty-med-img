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
    wandb.init(project="resnet50", config=vars(args))
    config = wandb.config

    # --- Get data loaders ---
    if args.dataset.lower() == "aptos2019":
        train_loader, val_loader, _, num_classes = get_aptos_loaders(batch_size=args.batch_size)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # --- Compute class weights ---
    labels = torch.tensor([y for _, y in train_loader.dataset])
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)

    # --- Model ---
    model = get_resnet50(num_classes=num_classes, weights=ResNet50_Weights.DEFAULT)
    model = model.to(device)

    # Freeze everything initially
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():  # Only FC trainable first
        param.requires_grad = True

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 7

    for epoch in range(config.epochs):
        # --- Progressive unfreezing schedule ---
        if epoch == 5:   # after 5 epochs
            for param in model.layer4.parameters():
                param.requires_grad = True
        if epoch == 10:  # after 10 epochs
            for param in model.layer3.parameters():
                param.requires_grad = True
        if epoch == 20:  # after 20 epochs
            for param in model.layer2.parameters():
                param.requires_grad = True

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

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

        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"Train loss: {epoch_loss:.4f} acc: {epoch_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        scheduler.step()

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
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"./outputs/models/model_ResNet50_{args.dataset}_{timestamp}_valacc{val_acc:.4f}.pth"
    torch.save(best_model_state, model_filename)
    wandb.save(model_filename)
    wandb.finish()
