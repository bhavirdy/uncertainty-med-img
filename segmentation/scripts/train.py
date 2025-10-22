import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb

from segmentation.models.unet import UNet, UNetEDL
from segmentation.data_loaders.isic2018_segmentation_data_loader import get_isic2018_segmentation_loaders
from segmentation.utils.segmentation_loss import combined_loss, evidential_segmentation_loss
from segmentation.utils.segmentation_metrics import dice_score, iou_score, pixel_accuracy

def train_epoch(model, train_loader, device, optimizer, epoch, args):
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_acc = 0

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
            loss = combined_loss(logits, masks, alpha=0.5)
            probs = torch.softmax(logits, dim=1)

        loss.backward()
        optimizer.step()

        # Compute metrics
        dice = dice_score(logits, masks)
        iou = iou_score(logits, masks)
        acc = pixel_accuracy(logits, masks)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou
        total_acc += acc

        # Print occasional updates
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} | Dice: {dice:.4f} | IoU: {iou:.4f} | Acc: {acc:.4f}"
            )

    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    avg_acc = total_acc / len(train_loader)

    return avg_loss, avg_dice, avg_iou, avg_acc

def validate_epoch(model, val_loader, device, epoch, args):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_acc = 0

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
                loss = combined_loss(logits, masks, alpha=0.5)

            # Compute metrics
            dice = dice_score(logits, masks)
            iou = iou_score(logits, masks)
            acc = pixel_accuracy(logits, masks)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            total_acc += acc

            # Print occasional updates
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                print(
                    f"Validation [{epoch}] Batch [{batch_idx + 1}/{len(val_loader)}] "
                    f"Loss: {loss.item():.4f} | Dice: {dice:.4f} | IoU: {iou:.4f} | Acc: {acc:.4f}"
                )

    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    avg_acc = total_acc / len(val_loader)

    return avg_loss, avg_dice, avg_iou, avg_acc

def train(model, train_loader, val_loader, device, args):
    # Initialize wandb
    wandb.init(
        project=f"unet-{args.dataset.lower()}{'-edl' if args.edl else ''}",    
        config=vars(args)
    )
    wandb.init(
        project=f"resnet50-{args.dataset.lower()}{'-edl' if args.edl else ''}",
        config=vars(args)
    )
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])
    
    best_dice = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        train_loss, train_dice, train_iou, train_acc = train_epoch(
            model, train_loader, device, optimizer, epoch, args
        )
        
        # Validation
        val_loss, val_dice, val_iou, val_acc = validate_epoch(
            model, val_loader, device, epoch, args
        )
        
        # Learning rate scheduling
        scheduler.step()
        
        # Logging
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'train_iou': train_iou,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
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
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')
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
        train_loader, val_loader, _, num_classes = get_isic2018_segmentation_loaders(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
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
