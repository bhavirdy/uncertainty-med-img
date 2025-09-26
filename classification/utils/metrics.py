import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss

def accuracy(preds, labels):
    """Compute accuracy"""
    preds = torch.argmax(preds, dim=1) if preds.ndim > 1 else preds
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def precision(preds, labels):
    """Compute precision"""
    preds = torch.argmax(preds, dim=1) if preds.ndim > 1 else preds
    return precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)

def recall(preds, labels):
    """Compute recall"""
    preds = torch.argmax(preds, dim=1) if preds.ndim > 1 else preds
    return recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)

def f1(preds, labels):
    """Compute F1 score"""
    preds = torch.argmax(preds, dim=1) if preds.ndim > 1 else preds
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

def auroc(uncertainty, errors):
    """AUROC for uncertainty vs. misclassification"""
    pass

def aupr(uncertainty, errors):
    """AUPR for uncertainty vs. misclassification"""
    pass

def aurc(uncertainty, errors):
    """Area under the risk-coverage curve"""
    pass

def fpr_at_95_tpr(uncertainty, errors):
    """False Positive Rate at 95% True Positive Rate"""
    pass
