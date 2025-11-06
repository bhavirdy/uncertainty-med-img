import numpy as np
from sklearn.metrics import log_loss

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