import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, roc_auc_score, average_precision_score

def accuracy(outputs, labels):
    """Compute accuracy"""
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def precision(outputs, labels):
    """Compute precision"""
    preds = torch.argmax(outputs, dim=1)
    return precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)

def recall(outputs, labels):
    """Compute recall"""
    preds = torch.argmax(outputs, dim=1)
    return recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)

def f1(outputs, labels):
    """Compute F1 score"""
    preds = torch.argmax(outputs, dim=1)
    return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)

def ece(probs, labels, n_bins=15):
    """Expected Calibration Error"""
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    ece_val = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins+1, device=probs.device)
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            ece_val += torch.abs(accuracies[mask].float().mean() - confidences[mask].mean()) * mask.float().mean()
    return ece_val.item()

def mce(probs, labels, n_bins=15):
    """Maximum Calibration Error"""
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins+1, device=probs.device)
    errors = []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            errors.append(torch.abs(accuracies[mask].float().mean() - confidences[mask].mean()).item())
    return max(errors) if errors else 0.0

def brier(probs, labels):
    """Brier Score"""
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    one_hot = np.eye(probs_np.shape[1])[labels_np]
    return np.mean(np.sum((probs_np - one_hot) ** 2, axis=1))

def nll(probs, labels):
    """Negative Log-Likelihood"""
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    return log_loss(labels_np, probs_np, labels=[i for i in range(probs_np.shape[1])])

def auroc(y_true, y_scores):
    """Compute AUROC for binary or multilabel (one-vs-rest)"""
    y_true_np = y_true.detach().cpu().numpy()
    y_scores_np = y_scores.detach().cpu().numpy()
    try:
        return roc_auc_score(y_true_np, y_scores_np, average='macro', multi_class='ovr')
    except ValueError:
        return float('nan')

def aupr(y_true, y_scores):
    """Compute area under precision-recall curve"""
    y_true_np = y_true.detach().cpu().numpy()
    y_scores_np = y_scores.detach().cpu().numpy()
    try:
        return average_precision_score(y_true_np, y_scores_np, average='macro')
    except ValueError:
        return float('nan')
