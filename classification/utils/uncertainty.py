import torch

def mcdo_predictions(model, inputs, n_samples=20):
    """
    Monte Carlo Dropout predictions.
    Runs T stochastic forward passes with dropout enabled.
    """
    model.train()  # enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = torch.softmax(model(inputs), dim=1)
            preds.append(out.unsqueeze(0))
    return torch.cat(preds, dim=0)  # [T, B, C]

def deep_ensemble_predictions(models, inputs):
    """Return predictions from a deep ensemble"""
    pass

def edl_predictions(model, inputs):
    """Return predictions and uncertainty from evidential deep learning"""
    pass

def predictive_mean(pred_samples):
    """Mean predictive distribution"""
    return pred_samples.mean(dim=0)

def predictive_variance(pred_samples):
    """Compute predictive variance"""
    return pred_samples.var(dim=0).mean(dim=1)

def predictive_entropy(probs):
    """Compute predictive entropy"""
    return -(probs * torch.log(probs + 1e-12)).sum(dim=1)