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
