import matplotlib.pyplot as plt
import torch
from sklearn.calibration import calibration_curve

def reliability_diagram(probs, labels, output_path, bins=15):
    num_classes = probs.shape[1]
    plt.figure(figsize=(8, 8))
    plt.plot([0,1], [0,1], '--', color='gray', label='Perfectly calibrated')
    
    for class_idx in range(num_classes):
        # One-vs-rest labels for the current class
        labels_bin = (labels == class_idx).astype(int)
        prob_class = probs[:, class_idx]
        
        prob_true, prob_pred = calibration_curve(labels_bin, prob_class, n_bins=bins, strategy='uniform')
        
        plt.plot(prob_pred, prob_true, marker='o', label=f'Class {class_idx}')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def predictive_entropy_histogram(probs, output_path, bins=30):
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1).cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(entropy, bins=bins, alpha=0.7, edgecolor='black')
    plt.xlabel('Predictive Entropy')
    plt.ylabel('Frequency')
    plt.title('Predictive Entropy Histogram')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
