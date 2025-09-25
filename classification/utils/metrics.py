from sklearn.metrics import precision_score, recall_score, f1_score
import torch

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
    pass

def brier_score(probs, labels):
    """Brier Score"""
    pass

def negative_log_likelihood(probs, labels):
    """Negative Log-Likelihood"""
    pass

def auroc(uncertainty, errors):
    """AUROC for uncertainty vs. misclassification"""
    pass

def aupr(uncertainty, errors):
    """AUPR for uncertainty vs. misclassification"""
    pass

def aurc(uncertainty, errors):
    """Area under the rejection curve"""
    pass

def fpr_at_95_tpr(uncertainty, errors):
    """False Positive Rate at 95% True Positive Rate"""
    pass
