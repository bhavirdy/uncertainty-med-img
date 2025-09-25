import torch

def mcdo_uncertainty(mc_predictions):
    """
    mc_predictions: tensor of shape [n_samples, batch_size, num_classes]
    Returns:
        mean: predictive mean
        variance: predictive variance (uncertainty)
    """
    mean = mc_predictions.mean(dim=0)
    variance = mc_predictions.var(dim=0)
    return mean, variance
