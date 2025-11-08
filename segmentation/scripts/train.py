import argparse
import os
import csv
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import JaccardIndex, F1Score, Accuracy

from segmentation.models.unet import UNet, UNetEDL
from segmentation.data_loaders.isic2018_segmentation_data_loader import get_isic2018_loaders
from segmentation.utils.evidential_loss import evidential_loss

def train(model, train_loader, val_loader, device, args):
    # --- Initialize wandb ---
    wandb.init(
        project=f"unet-{args.dataset.lower()}{'-edl' if args.edl else ''}",    
        config=vars(args)
    )

    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    train_dice_metric = F1Score(num_classes=args.num_classes, average='macro').to(device)
    train_iou_metric = JaccardIndex(num_classes=args.num_classes, average='macro').to(device)
    train_acc_metric = Accuracy(task="multiclass", num_classes=args.num_classes).to(device)
    val_dice_metric = F1Score(num_classes=args.num_classes, average='macro').to(device)
    val_iou_metric = JaccardIndex(num_classes=args.num_classes, average='macro').to(device)
    val_acc_metric = Accuracy(task="multiclass", num_classes=args.num_classes).to(device)
    
    best_dice = 0
    early_stop_counter = 0

    os.makedirs(args.output_dir, exist_ok=True)
    log_csv_path = os.path.join(args.output_dir, "train_log.csv")
    with open(log_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_dice', 'train_iou', 'train_acc', 'val_loss', 'val_dice', 'val_iou', 'val_acc'])
    
    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_dice_metric.reset()
        train_iou_metric.reset()
        train_acc_metric.reset()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)

            if args.edl:
                loss = evidential_loss(
                    alpha=outputs, 
                    target=labels,
                    num_classes=args.num_classes,
                    epoch=epoch,
                    annealing_epochs=args.annealing_epochs,
                    lambda_reg=args.lambda_reg
                )
            else:
                loss = nn.CrossEntropyLoss()(outputs, labels.long())

            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)

            train_dice_metric.update(preds, labels)
            train_iou_metric.update(preds, labels)
            train_acc_metric.update(preds, labels)
            train_loss += loss.item() * imgs.size(0)

        avg_train_dice = train_dice_metric.compute().item()
        avg_train_iou = train_iou_metric.compute().item()
        avg_train_acc = train_acc_metric.compute().item()
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_dice_metric.reset()
        val_iou_metric.reset()
        val_acc_metric.reset()

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)

                if args.edl:
                    loss = evidential_loss(
                        alpha=outputs, 
                        target=labels, 
                        num_classes=args.num_classes, 
                        epoch=epoch,
                        annealing_epochs=args.annealing_epochs,
                        lambda_reg=args.lambda_reg
                    )
                else:
                    loss = nn.CrossEntropyLoss()(outputs, labels.long())

                preds = torch.argmax(outputs, dim=1)

                val_dice_metric.update(preds, labels)
                val_iou_metric.update(preds, labels)
                val_acc_metric.update(preds, labels)
                val_loss += loss.item() * imgs.size(0)

        avg_val_dice = val_dice_metric.compute().item()
        avg_val_iou = val_iou_metric.compute().item()
        avg_val_acc = val_acc_metric.compute().item()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_dice)
        
        print(f"Epoch: {epoch+1}/{args.epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

        # --- Logging ---
        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_train_dice, avg_train_iou, avg_train_acc, avg_val_loss, avg_val_dice, avg_val_iou, avg_val_acc])

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_dice': avg_train_dice,
            'val_loss': avg_val_loss,
            'val_dice': avg_val_dice,
        })
        
        # --- Early stopping ---
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            early_stop_counter = 0
            save_model(model.state_dict(), args.output_dir, 'best_model.pth')
        else:
            early_stop_counter += 1   
            if early_stop_counter >= args.early_stop_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
    
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
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save logs and model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--early_stop_patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--edl', action='store_true', help='Use Evidential Deep Learning loss')
    parser.add_argument('--annealing_epochs', type=int, default=10, help='EDL annealing epochs')
    parser.add_argument('--lambda_reg', type=float, default=0.001, help='EDL lambda regularisation term')

    args = parser.parse_args()

    # --- Device setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- Data loaders ---
    if args.dataset.lower() == "isic2018":
        train_loader, val_loader, _ = get_isic2018_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # --- Model ---
    if args.edl:
        model = UNetEDL(n_channels=3, n_classes=args.num_classes, dropout_p=args.dropout)
    else:
        model = UNet(n_channels=3, n_classes=args.num_classes, dropout_p=args.dropout)

    # --- Train ---
    train(model, train_loader, val_loader, device, args)

if __name__ == "__main__":
    main()
