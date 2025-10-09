import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def reliability_diagram(outputs, labels, n_bins=15, output_path=""):
    """
    Reliability diagram that accepts raw logits.
    """
    probs = F.softmax(outputs, dim=1)
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins+1)
    acc_bin, conf_bin = [], []

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
    if output_path:
        plt.savefig(output_path)
    plt.close()

def predictive_entropy_histogram(outputs, bins=30, output_path=""):
    """
    Predictive entropy histogram that accepts raw logits.
    """
    probs = F.softmax(outputs, dim=1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1).cpu().numpy()

    plt.figure()
    plt.hist(entropy, bins=bins)
    plt.xlabel('Predictive Entropy')
    plt.ylabel('Number of samples')
    plt.title('Predictive Entropy Histogram')
    if output_path:
        plt.savefig(output_path)
    plt.close()

def reliability_diagram_from_probs(probs, labels, n_bins=15, output_path=""):
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
    plt.savefig(output_path)
    plt.close()

def predictive_entropy_histogram_from_probs(probs, bins=30, output_path=""):
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1).cpu().numpy()
    plt.figure()
    plt.hist(entropy, bins=bins)
    plt.xlabel('Predictive Entropy')
    plt.ylabel('Number of samples')
    plt.title('Predictive Entropy Histogram')
    plt.savefig(output_path)
    plt.close()
