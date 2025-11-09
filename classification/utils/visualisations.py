import matplotlib.pyplot as plt
import torch
from sklearn.calibration import calibration_curve

def plot_reliability_diagram(all_probs, all_labels, output_path, bins=15):
    num_classes = all_probs.shape[1]
    plt.figure(figsize=(8, 8))
    plt.plot([0,1], [0,1], '--', color='gray', label='Perfectly calibrated')
    
    for class_idx in range(num_classes):
        # One-vs-rest labels for the current class
        labels_bin = (all_labels == class_idx).int().numpy()
        prob_class = all_probs[:, class_idx].numpy()
        
        prob_true, prob_pred = calibration_curve(labels_bin, prob_class, n_bins=bins, strategy='uniform')
        
        plt.plot(prob_pred, prob_true, marker='o', label=f'Class {class_idx}')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_entropy_histogram(all_probs, output_path, bins=30):
    entropy = -(all_probs * torch.log(all_probs + 1e-12)).sum(dim=1).cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(entropy, bins=bins, alpha=0.7, edgecolor='black')
    plt.xlabel('Predictive Entropy')
    plt.ylabel('Frequency')
    plt.title('Predictive Entropy Histogram')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
