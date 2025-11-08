import matplotlib.pyplot as plt
import torch

def reliability_diagram(probs, labels, output_path, bins=15):
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels).float()
    bin_boundaries = torch.linspace(0, 1, bins+1)
    acc_bin = []
    conf_bin = []
    for i in range(bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            acc_bin.append(accuracies[mask].mean().item())
            conf_bin.append(confidences[mask].mean().item())
    plt.figure(figsize=(8, 8))
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.plot(conf_bin, acc_bin, marker='o')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.grid(True, alpha=0.3)
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
