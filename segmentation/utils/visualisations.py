import torch
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def reliability_diagram(probs, labels, output_path, bins=15):
    probs = probs.cpu().numpy()
    labels = labels.cpu().numpy()

    C, H, W = probs.shape
    num_classes = C
    probs_flat = probs.reshape(C, -1).T  # [N, C]
    labels_flat = labels.reshape(-1)     # [N]

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')

    for class_idx in range(num_classes):
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

def entropy_heatmap(probs, labels, output_path):
    probs = probs.cpu()
    labels = labels.cpu()

    uncertainty = -(probs * torch.log(probs + 1e-8)).sum(dim=0)  # [H, W]

    pred_mask = torch.argmax(probs, dim=0).numpy()
    labels_map = labels.numpy()
    uncertainty_map = uncertainty.numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(labels_map, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    im = plt.imshow(uncertainty_map, cmap='hot', interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f'Uncertainty')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
