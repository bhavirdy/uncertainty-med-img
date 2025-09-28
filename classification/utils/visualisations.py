import matplotlib.pyplot as plt
import torch

def reliability_diagram(probs, labels, n_bins=15):
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins+1)
    acc_bin = []
    conf_bin = []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            acc_bin.append(accuracies[mask].mean().item())
            conf_bin.append(confidences[mask].mean().item())
    plt.figure()
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.plot(conf_bin, acc_bin, marker='o')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.show()

def predictive_entropy_histogram(probs, bins=30):
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1).cpu().numpy()
    plt.figure()
    plt.hist(entropy, bins=bins)
    plt.xlabel('Predictive Entropy')
    plt.ylabel('Number of samples')
    plt.title('Predictive Entropy Histogram')
    plt.show()

def confidence_accuracy_plot(probs, labels, n_bins=15):
    confidences, predictions = torch.max(probs, 1)
    accuracies = (predictions == labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins+1)
    acc_bin = []
    conf_bin = []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            acc_bin.append(accuracies[mask].mean().item())
            conf_bin.append(confidences[mask].mean().item())
    plt.figure()
    plt.plot(conf_bin, acc_bin, marker='o')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Confidence vs Accuracy')
    plt.show()
