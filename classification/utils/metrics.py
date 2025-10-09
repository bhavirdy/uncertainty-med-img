import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, roc_auc_score, average_precision_score

# ------------------------------
# Performance metrics
# ------------------------------

def accuracy(probs, labels, per_class=False):
    preds = torch.argmax(probs, dim=1)
    correct = (preds == labels)

    if not per_class:
        return correct.float().mean().item()

    # Per-class accuracy
    n_classes = probs.size(1)
    accs = []
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum().item() > 0:
            accs.append(correct[mask].float().mean().item())
        else:
            accs.append(float('nan'))
    return accs

def precision(probs, labels, per_class=False):
    preds = torch.argmax(probs, dim=1)
    average = None if per_class else 'macro'
    return precision_score(labels.cpu().numpy(), preds.cpu().numpy(),
                           average=average, zero_division=0)

def recall(probs, labels, per_class=False):
    preds = torch.argmax(probs, dim=1)
    average = None if per_class else 'macro'
    return recall_score(labels.cpu().numpy(), preds.cpu().numpy(),
                        average=average, zero_division=0)

def f1(probs, labels, per_class=False):
    preds = torch.argmax(probs, dim=1)
    average = None if per_class else 'macro'
    return f1_score(labels.cpu().numpy(), preds.cpu().numpy(),
                    average=average, zero_division=0)

def auroc(probs, labels, per_class=False):
    y_true = F.one_hot(labels, num_classes=probs.size(1)).cpu().numpy()
    y_scores = probs.cpu().numpy()
    try:
        if per_class:
            return roc_auc_score(y_true, y_scores, average=None, multi_class='ovr').tolist()
        return roc_auc_score(y_true, y_scores, average='macro', multi_class='ovr')
    except ValueError:
        return float('nan') if not per_class else [float('nan')] * probs.size(1)

def aupr(probs, labels, per_class=False):
    y_true = F.one_hot(labels, num_classes=probs.size(1)).cpu().numpy()
    y_scores = probs.cpu().numpy()
    try:
        if per_class:
            return average_precision_score(y_true, y_scores, average=None).tolist()
        return average_precision_score(y_true, y_scores, average='macro')
    except ValueError:
        return float('nan') if not per_class else [float('nan')] * probs.size(1)

# ------------------------------
# Uncertainty / calibration metrics
# ------------------------------

def ece(probs, labels, n_bins=15):
    """Expected Calibration Error"""
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    ece_val = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            ece_val += torch.abs(accuracies[mask].float().mean() - confidences[mask].mean()) * mask.float().mean()
    return ece_val.item()

def mce(probs, labels, n_bins=15):
    """Maximum Calibration Error"""
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    errors = []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            errors.append(torch.abs(accuracies[mask].float().mean() - confidences[mask].mean()).item())
    return max(errors) if errors else 0.0

def brier(probs, labels):
    """Brier score for probability vectors"""
    probs_np = probs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    one_hot = np.eye(probs_np.shape[1])[labels_np]
    return float(np.mean(np.sum((probs_np - one_hot) ** 2, axis=1)))

def nll(probs, labels):
    """Negative log-likelihood"""
    probs_np = probs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    return float(log_loss(labels_np, probs_np, labels=[i for i in range(probs_np.shape[1])]))
