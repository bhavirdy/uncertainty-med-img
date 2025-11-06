import numpy as np
from sklearn.metrics import log_loss

def nll(probs, labels):
    """Negative log-likelihood for segmentation"""
    probs_flat = probs.reshape(probs.size(0), probs.size(1), -1)  # [B, C, H*W]
    labels_flat = labels.reshape(labels.size(0), -1)  # [B, H*W]
    
    # Flatten to 1D
    probs_flat = probs_flat.permute(0, 2, 1).contiguous().reshape(-1, probs.size(1))  # [B*H*W, C]
    labels_flat = labels_flat.reshape(-1)  # [B*H*W]
    
    probs_np = probs_flat.cpu().numpy()
    labels_np = labels_flat.cpu().numpy()
    
    return float(log_loss(labels_np, probs_np, labels=[0, 1]))

def brier(probs, labels):
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
