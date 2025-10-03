import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, roc_auc_score, average_precision_score

# --- Model evaluation --- 

def accuracy(outputs, labels, per_class=False):
    """Compute accuracy"""
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == labels)

    if not per_class:
        return correct.sum().item() / labels.size(0)

    # Per-class accuracy
    n_classes = outputs.size(1)
    accs = []
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum().item() > 0:
            accs.append(correct[mask].float().mean().item())
        else:
            accs.append(float('nan'))
    return accs

def precision(outputs, labels, per_class=False):
    preds = torch.argmax(outputs, dim=1)
    average = None if per_class else 'macro'
    return precision_score(labels.cpu().numpy(), preds.cpu().numpy(),
                           average=average, zero_division=0)

def recall(outputs, labels, per_class=False):
    preds = torch.argmax(outputs, dim=1)
    average = None if per_class else 'macro'
    return recall_score(labels.cpu().numpy(), preds.cpu().numpy(),
                        average=average, zero_division=0)

def f1(outputs, labels, per_class=False):
    preds = torch.argmax(outputs, dim=1)
    average = None if per_class else 'macro'
    return f1_score(labels.cpu().numpy(), preds.cpu().numpy(),
                    average=average, zero_division=0)

def auroc(outputs, labels, per_class=False):
    probs = F.softmax(outputs, dim=1)
    y_true_oh = F.one_hot(labels, num_classes=probs.size(1))

    y_true_np = y_true_oh.detach().cpu().numpy()
    y_scores_np = probs.detach().cpu().numpy()
    try:
        if per_class:
            return roc_auc_score(y_true_np, y_scores_np, average=None,
                                 multi_class='ovr').tolist()
        return roc_auc_score(y_true_np, y_scores_np, average='macro',
                             multi_class='ovr')
    except ValueError:
        return float('nan') if not per_class else [float('nan')] * probs.size(1)

def aupr(outputs, labels, per_class=False):
    probs = F.softmax(outputs, dim=1)
    y_true_oh = F.one_hot(labels, num_classes=probs.size(1))

    y_true_np = y_true_oh.detach().cpu().numpy()
    y_scores_np = probs.detach().cpu().numpy()
    try:
        if per_class:
            return average_precision_score(y_true_np, y_scores_np,
                                           average=None).tolist()
        return average_precision_score(y_true_np, y_scores_np,
                                       average='macro')
    except ValueError:
        return float('nan') if not per_class else [float('nan')] * probs.size(1)

# --- Uncertainty evaluation ---

def ece(outputs, labels, n_bins=15):
    """Expected Calibration Error (ECE)"""
    probs = F.softmax(outputs, dim=1)
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)

    ece_val = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins+1, device=probs.device)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            ece_val += torch.abs(accuracies[mask].float().mean() - confidences[mask].mean()) * mask.float().mean()

    return ece_val.item()

def mce(outputs, labels, n_bins=15):
    """Maximum Calibration Error (MCE)"""
    probs = F.softmax(outputs, dim=1)
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins+1, device=probs.device)

    errors = []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            errors.append(torch.abs(accuracies[mask].float().mean() - confidences[mask].mean()).item())

    return max(errors) if errors else 0.0

def brier(outputs, labels):
    """Brier Score"""
    probs = F.softmax(outputs, dim=1)
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    one_hot = np.eye(probs_np.shape[1])[labels_np]
    return float(np.mean(np.sum((probs_np - one_hot) ** 2, axis=1)))

def nll(outputs, labels):
    """Negative Log-Likelihood (NLL)"""
    probs = F.softmax(outputs, dim=1)
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    return float(log_loss(labels_np, probs_np, labels=[i for i in range(probs_np.shape[1])]))

def ece_from_probs(probs, labels, n_bins=15):
    """ECE with already averaged probabilities"""
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)

    ece_val = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins+1, device=probs.device)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            ece_val += torch.abs(accuracies[mask].float().mean() -
                                 confidences[mask].mean()) * mask.float().mean()
    return ece_val.item()

def mce_from_probs(probs, labels, n_bins=15):
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins+1, device=probs.device)

    errors = []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            errors.append(torch.abs(accuracies[mask].float().mean() -
                                    confidences[mask].mean()).item())
    return max(errors) if errors else 0.0

def brier_from_probs(probs, labels):
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    one_hot = np.eye(probs_np.shape[1])[labels_np]
    return float(np.mean(np.sum((probs_np - one_hot) ** 2, axis=1)))

def nll_from_probs(probs, labels):
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    return float(log_loss(labels_np, probs_np, labels=[i for i in range(probs_np.shape[1])]))
