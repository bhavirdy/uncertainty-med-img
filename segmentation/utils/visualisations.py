import torch
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_reliability_diagram(all_probs, all_labels, output_path, bins=15):
    # Convert to NumPy
    all_probs = all_probs.cpu().numpy()
    all_labels = all_labels.cpu().numpy()

    B, C, H, W = all_probs.shape

    # Flatten batch and spatial dimensions
    probs_flat = all_probs.transpose(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
    labels_flat = all_labels.reshape(-1)                          # [B*H*W]

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')

    for class_idx in range(C):
        labels_bin = (labels_flat == class_idx).astype(int)
        prob_class = probs_flat[:, class_idx]

        prob_true, prob_pred = calibration_curve(labels_bin, prob_class, n_bins=bins, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker='o', label=f'Class {class_idx}')

    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_entropy_heatmaps(all_probs, all_labels, output_path, num_samples=9, seed=42):
    torch.manual_seed(seed)
    N = all_probs.shape[0]
    num_samples = min(num_samples, N)

    indices = torch.randperm(N)[:num_samples]

    n_cols = 3  # Ground Truth | Prediction | Uncertainty
    n_rows = num_samples
    plt.figure(figsize=(n_cols * 4, n_rows * 4))

    for i, idx in enumerate(indices):
        probs_img = all_probs[idx]       # [C, H, W]
        labels_img = all_labels[idx]     # [H, W]

        # Compute entropy per pixel
        uncertainty = -(probs_img * torch.log(probs_img + 1e-8)).sum(dim=0)  # [H, W]

        # Predicted mask
        pred_mask = torch.argmax(probs_img, dim=0)

        # Convert to NumPy for plotting
        labels_map = labels_img.cpu().numpy()
        pred_map = pred_mask.cpu().numpy()
        uncertainty_map = uncertainty.cpu().numpy()

        # Ground Truth
        plt.subplot(n_rows, n_cols, i * n_cols + 1)
        plt.imshow(labels_map, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        # Prediction
        plt.subplot(n_rows, n_cols, i * n_cols + 2)
        plt.imshow(pred_map, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        # Uncertainty
        plt.subplot(n_rows, n_cols, i * n_cols + 3)
        im = plt.imshow(uncertainty_map, cmap='hot', interpolation='nearest')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title('Uncertainty')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
