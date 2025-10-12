import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for segmentation"""
    pred = torch.softmax(pred, dim=1)
    pred = pred[:, 1]  # Get foreground class
    
    target = target.float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def dice_score(pred, target, smooth=1e-6):
    """Dice score for evaluation"""
    pred = torch.softmax(pred, dim=1)
    pred = pred[:, 1]  # Get foreground class
    
    target = target.float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice

def combined_loss(pred, target, alpha=0.5):
    """Combined Dice + Cross-entropy loss"""
    ce_loss = F.cross_entropy(pred, target.long())
    dice_loss_val = dice_loss(pred, target)
    return alpha * ce_loss + (1 - alpha) * dice_loss_val

def evidential_segmentation_loss(alpha, target, num_classes, epoch, annealing_epochs=10, lambda_reg=0.01):
    """
    Evidential loss adapted for segmentation
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / S
    
    # Convert target to one-hot
    target_onehot = F.one_hot(target.long(), num_classes=num_classes).float()
    target_onehot = target_onehot.permute(0, 3, 1, 2)  # [B, C, H, W]
    
    # Data (MSE) term
    mse = torch.sum((target_onehot - probs) ** 2, dim=1)
    var = torch.sum(probs * (1 - probs) / (S + 1), dim=1)
    loss_data = mse + var
    
    # KL divergence
    def KL(alpha):
        K = alpha.shape[1]
        beta = torch.ones((1, K, 1, 1), device=alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_beta = torch.lgamma(S_beta) - torch.sum(torch.lgamma(beta), dim=1, keepdim=True)
        digamma_diff = torch.digamma(alpha) - torch.digamma(S_alpha)
        kl = torch.sum((alpha - beta) * digamma_diff, dim=1, keepdim=True) + lnB_alpha - lnB_beta
        return kl.squeeze(1)
    
    kl_div = KL(alpha)
    
    # Annealing
    annealing_coef = min(1.0, epoch / annealing_epochs)
    
    loss = loss_data + lambda_reg * annealing_coef * kl_div
    return torch.mean(loss)
