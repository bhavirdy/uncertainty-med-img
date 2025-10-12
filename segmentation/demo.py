#!/usr/bin/env python3
"""
Example script demonstrating the segmentation uncertainty quantification pipeline.
This script shows how to train and evaluate models using different uncertainty methods.
"""

import os
import torch
import argparse
from segmentation.models.unet import UNet, UNetEDL
from segmentation.data_loaders.isic2018_segmentation_data_loader import get_isic2018_segmentation_loaders
from segmentation.utils.segmentation_metrics import dice_score, iou_score, pixel_accuracy
from segmentation.utils.segmentation_uncertainty import mcdo_segmentation_predictions, edl_segmentation_predictions

def demo_training():
    """Demonstrate training a segmentation model"""
    print("=== Segmentation Training Demo ===")
    
    # Load data
    train_loader, val_loader, test_loader, num_classes = get_isic2018_segmentation_loaders(
        batch_size=4, num_workers=2
    )
    
    print(f"Dataset loaded: {num_classes} classes")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=num_classes, dropout_p=0.5)
    model = model.to(device)
    
    print(f"Model created: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            
            print(f"Input shape: {images.shape}")
            print(f"Output shape: {logits.shape}")
            print(f"Mask shape: {masks.shape}")
            break
    
    print("Training demo completed!\n")

def demo_uncertainty_methods():
    """Demonstrate different uncertainty estimation methods"""
    print("=== Uncertainty Methods Demo ===")
    
    # Load data and model
    _, _, test_loader, num_classes = get_isic2018_segmentation_loaders(
        batch_size=2, num_workers=2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test deterministic model
    print("1. Deterministic Model:")
    det_model = UNet(n_channels=3, n_classes=num_classes, dropout_p=0.5)
    det_model = det_model.to(device)
    det_model.eval()
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            logits = det_model(images)
            probs = torch.softmax(logits, dim=1)
            
            dice = dice_score(logits, masks)
            iou = iou_score(logits, masks)
            acc = pixel_accuracy(logits, masks)
            
            print(f"   Dice: {dice:.4f}, IoU: {iou:.4f}, Accuracy: {acc:.4f}")
            break
    
    # Test MCDO
    print("2. Monte Carlo Dropout:")
    mcdo_model = UNet(n_channels=3, n_classes=num_classes, dropout_p=0.5)
    mcdo_model = mcdo_model.to(device)
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            # Get MC samples
            pred_samples = mcdo_segmentation_predictions(mcdo_model, images, n_samples=10)
            probs = pred_samples.mean(dim=0)
            uncertainty = pred_samples.var(dim=0)
            
            dice = dice_score(torch.log(probs + 1e-8), masks)
            iou = iou_score(torch.log(probs + 1e-8), masks)
            acc = pixel_accuracy(torch.log(probs + 1e-8), masks)
            
            print(f"   Dice: {dice:.4f}, IoU: {iou:.4f}, Accuracy: {acc:.4f}")
            print(f"   Uncertainty shape: {uncertainty.shape}")
            break
    
    # Test EDL
    print("3. Evidential Deep Learning:")
    edl_model = UNetEDL(n_channels=3, n_classes=num_classes, dropout_p=0.5)
    edl_model = edl_model.to(device)
    edl_model.eval()
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            alpha = edl_model(images)
            probs = alpha / alpha.sum(dim=1, keepdim=True)
            
            dice = dice_score(torch.log(probs + 1e-8), masks)
            iou = iou_score(torch.log(probs + 1e-8), masks)
            acc = pixel_accuracy(torch.log(probs + 1e-8), masks)
            
            print(f"   Dice: {dice:.4f}, IoU: {iou:.4f}, Accuracy: {acc:.4f}")
            print(f"   Alpha shape: {alpha.shape}")
            break
    
    print("Uncertainty methods demo completed!\n")

def demo_metrics():
    """Demonstrate segmentation metrics"""
    print("=== Metrics Demo ===")
    
    # Create dummy data
    batch_size, num_classes, height, width = 2, 2, 64, 64
    
    # Random predictions
    logits = torch.randn(batch_size, num_classes, height, width)
    probs = torch.softmax(logits, dim=1)
    
    # Random ground truth
    masks = torch.randint(0, num_classes, (batch_size, height, width)).float()
    
    # Compute metrics
    dice = dice_score(logits, masks)
    iou = iou_score(logits, masks)
    acc = pixel_accuracy(logits, masks)
    
    print(f"Dice Score: {dice:.4f}")
    print(f"IoU Score: {iou:.4f}")
    print(f"Pixel Accuracy: {acc:.4f}")
    
    print("Metrics demo completed!\n")

def main():
    parser = argparse.ArgumentParser(description="Segmentation uncertainty demo")
    parser.add_argument("--demo", type=str, choices=["training", "uncertainty", "metrics", "all"], 
                       default="all", help="Which demo to run")
    args = parser.parse_args()
    
    print("Segmentation Uncertainty Quantification Demo")
    print("=" * 50)
    
    if args.demo in ["training", "all"]:
        demo_training()
    
    if args.demo in ["uncertainty", "all"]:
        demo_uncertainty_methods()
    
    if args.demo in ["metrics", "all"]:
        demo_metrics()
    
    print("All demos completed!")

if __name__ == "__main__":
    main()
