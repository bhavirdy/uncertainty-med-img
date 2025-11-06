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
    return torch.cat(preds, dim=0)

def predictive_mean(pred_samples):
    """Mean predictive distribution"""
    return pred_samples.mean(dim=0)

def predictive_variance(pred_samples):
    """Compute predictive variance"""
    return pred_samples.var(dim=0)

def predictive_entropy(pred_samples):
    """Compute predictive entropy"""
    mean_pred = pred_samples.mean(dim=0)
    entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=1)
    return entropy
