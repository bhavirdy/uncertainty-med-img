import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from segmentation.models.unet import UNet, UNetEDL
from segmentation.data_loaders.isic2018_segmentation_data_loader import get_isic2018_loaders
from segmentation.utils.evidential_loss import evidential_segmentation_loss
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassAccuracy

def train_epoch(model, train_loader, device, optimizer, epoch, args, dice_metric, iou_metric, acc_metric):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        if args.edl:
            alpha = model(images)
            loss = evidential_segmentation_loss(
                alpha, masks, args.num_classes, epoch,
                annealing_epochs=args.annealing_epochs
            )

            # Convert alpha to probabilities for metrics
            S = alpha.sum(dim=1, keepdim=True)
            probs = alpha / S
            logits = torch.log(probs + 1e-8)
        else:
            logits = model(images)
            loss = criterion(logits, masks.long())

        loss.backward()
        optimizer.step()

        # Convert logits to predicted masks
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        dice_metric.update(preds, masks)
        iou_metric.update(preds, masks)
        acc_metric.update(preds, masks)

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_dice = dice_metric.compute().item()
    avg_iou = iou_metric.compute().item()
    avg_acc = acc_metric.compute().item()

    # Reset for next epoch
    dice_metric.reset()
    iou_metric.reset()
    acc_metric.reset()

    return avg_loss, avg_dice, avg_iou, avg_acc

def validate_epoch(model, val_loader, device, epoch, args, dice_metric, iou_metric, acc_metric):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)

            if args.edl:
                alpha = model(images)
                loss = evidential_segmentation_loss(
                    alpha, masks, args.num_classes, epoch,
                    annealing_epochs=args.annealing_epochs
                )
                # Convert alpha to probabilities for metrics
                S = alpha.sum(dim=1, keepdim=True)
                probs = alpha / S
                logits = torch.log(probs + 1e-8)
            else:
                logits = model(images)
                loss = criterion(logits, masks.long())

            # Convert logits to predicted masks
            preds = torch.argmax(logits, dim=1)

            # Update metrics
            dice_metric.update(preds, masks)
            iou_metric.update(preds, masks)
            acc_metric.update(preds, masks)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    avg_dice = dice_metric.compute().item()
    avg_iou = iou_metric.compute().item()
    avg_acc = acc_metric.compute().item()

    # Reset for next epoch
    dice_metric.reset()
    iou_metric.reset()
    acc_metric.reset()

    return avg_loss, avg_dice, avg_iou, avg_acc

def train(model, train_loader, val_loader, device, args):
    # Initialize wandb
    wandb.init(
        project=f"unet-{args.dataset.lower()}{'-edl' if args.edl else ''}",    
        config=vars(args)
    )
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler - ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    dice_metric = MulticlassF1Score(num_classes=args.num_classes, average='macro').to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=args.num_classes, average='macro').to(device)
    acc_metric = MulticlassAccuracy(num_classes=args.num_classes, average='micro').to(device)
    
    best_dice = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        train_loss, train_dice, train_iou, train_acc = train_epoch(
            model, train_loader, device, optimizer, epoch, args, dice_metric, iou_metric, acc_metric
        )
        
        # Validation
        val_loss, val_dice, val_iou, val_acc = validate_epoch(
            model, val_loader, device, epoch, args, dice_metric, iou_metric, acc_metric
        )
        
        # Learning rate scheduling
        scheduler.step(val_dice)
        
        # Logging
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_loss': val_loss,
            'val_dice': val_dice,
        })
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        
        # Early stopping
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            save_model(model.state_dict(), args.output_dir, 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= args.early_stop_patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # Save final model
    save_model(model.state_dict(), args.output_dir, 'final_model.pth')
    wandb.finish()

def save_model(model_state, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, filename)
    torch.save(model_state, model_path)
    print(f'Saved model to {model_path}')

def main():
    parser = argparse.ArgumentParser(description="Train U-Net for segmentation")

    # --- Arguments ---
    parser.add_argument('--dataset', type=str, required=True, choices=['isic2018'], help='Dataset name')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save logs and model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--early_stop_patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--edl', action='store_true', help='Use Evidential Deep Learning loss')
    parser.add_argument('--annealing_epochs', type=int, default=10, help='EDL annealing epochs')
    parser.add_argument('--timestamp', type=str, default='', help='Timestamp for run identification')

    args = parser.parse_args()

    # --- Device setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- Data loaders ---
    if args.dataset.lower() == "isic2018":
        train_loader, val_loader, _, num_classes = get_isic2018_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    args.num_classes = num_classes

    # --- Model ---
    if args.edl:
        model = UNetEDL(n_channels=3, n_classes=num_classes, dropout_p=args.dropout)
    else:
        model = UNet(n_channels=3, n_classes=num_classes, dropout_p=args.dropout)

    model = model.to(device)
    print(f'Model: {model.__class__.__name__}')
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # --- Training ---
    train(model, train_loader, val_loader, device, args)

if __name__ == "__main__":
    main()
