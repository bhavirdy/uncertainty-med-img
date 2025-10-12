import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.calibration import calibration_curve

def reliability_diagram_from_probs(probs, labels, output_path, n_bins=15):
    """
    Generate reliability diagram for segmentation
    """
    # Flatten spatial dimensions
    probs_flat = probs.view(probs.size(0), probs.size(1), -1)  # [B, C, H*W]
    labels_flat = labels.view(labels.size(0), -1)  # [B, H*W]
    
    # Get foreground class probabilities
    fg_probs = probs_flat[:, 1]  # [B, H*W]
    fg_labels = labels_flat.float()  # [B, H*W]
    
    # Flatten to 1D
    fg_probs_flat = fg_probs.view(-1).cpu().numpy()  # [B*H*W]
    fg_labels_flat = fg_labels.view(-1).cpu().numpy()  # [B*H*W]
    
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        fg_labels_flat, fg_probs_flat, n_bins=n_bins
    )
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(mean_predicted_value, fraction_of_positives, 'ro-', label='Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def predictive_entropy_histogram_from_probs(probs, labels, output_path):
    """
    Generate predictive entropy histogram for segmentation
    """
    # Flatten spatial dimensions
    probs_flat = probs.view(probs.size(0), probs.size(1), -1)  # [B, C, H*W]
    
    # Compute entropy for each pixel
    entropy = -torch.sum(probs_flat * torch.log(probs_flat + 1e-8), dim=1)  # [B, H*W]
    entropy_flat = entropy.view(-1).cpu().numpy()  # [B*H*W]
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(entropy_flat, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Predictive Entropy')
    plt.ylabel('Frequency')
    plt.title('Predictive Entropy Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def uncertainty_heatmap(probs, labels, output_path, method='entropy'):
    """
    Generate uncertainty heatmap for segmentation
    """
    # Flatten spatial dimensions
    probs_flat = probs.view(probs.size(0), probs.size(1), -1)  # [B, C, H*W]
    
    if method == 'entropy':
        # Compute entropy for each pixel
        uncertainty = -torch.sum(probs_flat * torch.log(probs_flat + 1e-8), dim=1)  # [B, H*W]
    elif method == 'max_prob':
        # Use max probability as uncertainty (lower max prob = higher uncertainty)
        uncertainty = 1 - torch.max(probs_flat, dim=1)[0]  # [B, H*W]
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")
    
    # Reshape back to spatial dimensions
    batch_size = probs.size(0)
    height = width = int(np.sqrt(probs_flat.size(2)))
    uncertainty = uncertainty.view(batch_size, height, width)  # [B, H, W]
    
    # Plot heatmap for first image in batch
    plt.figure(figsize=(12, 4))
    
    # Original image (if available)
    plt.subplot(1, 3, 1)
    plt.imshow(labels[0].cpu().numpy(), cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 3, 2)
    pred_mask = torch.argmax(probs[0], dim=0).cpu().numpy()
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    
    # Uncertainty heatmap
    plt.subplot(1, 3, 3)
    uncertainty_map = uncertainty[0].cpu().numpy()
    im = plt.imshow(uncertainty_map, cmap='hot', interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f'Uncertainty ({method})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def aleatoric_epistemic_heatmap(aleatoric, epistemic, labels, output_path):
    """
    Generate separate aleatoric and epistemic uncertainty heatmaps
    """
    # Flatten spatial dimensions
    aleatoric_flat = aleatoric.view(aleatoric.size(0), aleatoric.size(1), -1)  # [B, C, H*W]
    epistemic_flat = epistemic.view(epistemic.size(0), epistemic.size(1), -1)  # [B, C, H*W]
    
    # Get foreground class uncertainty
    aleatoric_fg = aleatoric_flat[:, 1]  # [B, H*W]
    epistemic_fg = epistemic_flat[:, 1]  # [B, H*W]
    
    # Reshape back to spatial dimensions
    batch_size = aleatoric.size(0)
    height = width = int(np.sqrt(aleatoric_flat.size(2)))
    aleatoric_fg = aleatoric_fg.view(batch_size, height, width)  # [B, H, W]
    epistemic_fg = epistemic_fg.view(batch_size, height, width)  # [B, H, W]
    
    # Plot heatmaps
    plt.figure(figsize=(15, 5))
    
    # Ground truth
    plt.subplot(1, 4, 1)
    plt.imshow(labels[0].cpu().numpy(), cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Aleatoric uncertainty
    plt.subplot(1, 4, 2)
    aleatoric_map = aleatoric_fg[0].cpu().numpy()
    im1 = plt.imshow(aleatoric_map, cmap='Blues', interpolation='nearest')
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.title('Aleatoric Uncertainty')
    plt.axis('off')
    
    # Epistemic uncertainty
    plt.subplot(1, 4, 3)
    epistemic_map = epistemic_fg[0].cpu().numpy()
    im2 = plt.imshow(epistemic_map, cmap='Reds', interpolation='nearest')
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.title('Epistemic Uncertainty')
    plt.axis('off')
    
    # Total uncertainty
    plt.subplot(1, 4, 4)
    total_map = aleatoric_map + epistemic_map
    im3 = plt.imshow(total_map, cmap='hot', interpolation='nearest')
    plt.colorbar(im3, fraction=0.046, pad=0.04)
    plt.title('Total Uncertainty')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
