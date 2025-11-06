import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import log_loss

def dice_score(pred, target, smooth=1e-6):
    """Dice score for segmentation"""
    pred = torch.softmax(pred, dim=1)
    pred = pred[:, 1]  # Get foreground class
    
    target = target.float()
    
    # Flatten spatial dimensions
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()

def iou_score(pred, target, smooth=1e-6):
    """Intersection over Union (IoU) score"""
    pred = torch.softmax(pred, dim=1)
    pred = pred[:, 1]  # Get foreground class
    
    target = target.float()
    
    # Flatten spatial dimensions
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()

def pixel_accuracy(pred, target):
    """Pixel-wise accuracy"""
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target.long()).float()
    return correct.mean().item()

def segmentation_ece(probs, labels, n_bins=15):
    """Expected Calibration Error for segmentation"""
    # Flatten spatial dimensions
    probs_flat = probs.reshape(probs.size(0), probs.size(1), -1)  # [B, C, H*W]
    labels_flat = labels.reshape(labels.size(0), -1)  # [B, H*W]
    
    # Get foreground class probabilities
    if probs.shape[1] == 1:
        fg_probs = probs_flat[:, 0]
    else:
        fg_probs = probs_flat[:, 1]
    fg_labels = labels_flat.float()  # [B, H*W]
    
    # Flatten to 1D
    fg_probs_flat = fg_probs.reshape(-1)  # [B*H*W]
    fg_labels_flat = fg_labels.reshape(-1)  # [B*H*W]
    
    confidences, predictions = torch.max(torch.stack([1-fg_probs_flat, fg_probs_flat], dim=1), dim=1)
    accuracies = predictions.eq(fg_labels_flat.long())
    
    ece_val = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            ece_val += torch.abs(accuracies[mask].float().mean() - confidences[mask].mean()) * mask.float().mean()
    
    return ece_val.item()

def segmentation_mce(probs, labels, n_bins=15):
    """Maximum Calibration Error for segmentation"""
    # Flatten spatial dimensions
    probs_flat = probs.reshape(probs.size(0), probs.size(1), -1)  # [B, C, H*W]
    labels_flat = labels.reshape(labels.size(0), -1)  # [B, H*W]
    
    # Get foreground class probabilities
    if probs.shape[1] == 1:
        fg_probs = probs_flat[:, 0]
    else:
        fg_probs = probs_flat[:, 1]
    fg_labels = labels_flat.float()  # [B, H*W]
    
    # Flatten to 1D
    fg_probs_flat = fg_probs.reshape(-1)  # [B*H*W]
    fg_labels_flat = fg_labels.reshape(-1)  # [B*H*W]
    
    confidences, predictions = torch.max(torch.stack([1-fg_probs_flat, fg_probs_flat], dim=1), dim=1)
    accuracies = predictions.eq(fg_labels_flat.long())
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    errors = []
    
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            errors.append(torch.abs(accuracies[mask].float().mean() - confidences[mask].mean()).item())
    
    return max(errors) if errors else 0.0

def segmentation_nll(probs, labels):
    """Negative log-likelihood for segmentation"""
    probs_flat = probs.reshape(probs.size(0), probs.size(1), -1)  # [B, C, H*W]
    labels_flat = labels.reshape(labels.size(0), -1)  # [B, H*W]
    
    # Flatten to 1D
    probs_flat = probs_flat.permute(0, 2, 1).contiguous().reshape(-1, probs.size(1))  # [B*H*W, C]
    labels_flat = labels_flat.reshape(-1)  # [B*H*W]
    
    probs_np = probs_flat.cpu().numpy()
    labels_np = labels_flat.cpu().numpy()
    
    return float(log_loss(labels_np, probs_np, labels=[0, 1]))

def segmentation_brier(probs, labels):
    """Brier score for segmentation"""
    probs_flat = probs.reshape(probs.size(0), probs.size(1), -1)  # [B, C, H*W]
    labels_flat = labels.reshape(labels.size(0), -1)  # [B, H*W]
    
    # Flatten to 1D
    probs_flat = probs_flat.permute(0, 2, 1).contiguous().reshape(-1, probs.size(1))  # [B*H*W, C]
    labels_flat = labels_flat.reshape(-1)  # [B*H*W]
    
    probs_np = probs_flat.cpu().numpy()
    labels_np = labels_flat.cpu().numpy()
    
    # Convert to one-hot
    one_hot = np.eye(probs_np.shape[1])[labels_np]
    
    return float(np.mean(np.sum((probs_np - one_hot) ** 2, axis=1)))
