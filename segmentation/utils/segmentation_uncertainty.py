import torch
import torch.nn.functional as F

def mcdo_segmentation_predictions(model, inputs, n_samples=20):
    """
    Monte Carlo Dropout predictions for segmentation.
    Runs T stochastic forward passes with dropout enabled.
    """
    model.train()  # enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = torch.softmax(model(inputs), dim=1)
            preds.append(out.unsqueeze(0))
    return torch.cat(preds, dim=0)  # [S, B, C, H, W]

def segmentation_predictive_mean(pred_samples):
    """Mean predictive distribution"""
    return pred_samples.mean(dim=0)

def segmentation_predictive_variance(pred_samples):
    """Compute predictive variance"""
    return pred_samples.var(dim=0)

def segmentation_predictive_entropy(pred_samples):
    """Compute predictive entropy"""
    mean_pred = pred_samples.mean(dim=0)
    entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=1)
    return entropy

def segmentation_aleatoric_uncertainty(pred_samples):
    """Compute aleatoric uncertainty (data uncertainty)"""
    mean_pred = pred_samples.mean(dim=0)
    aleatoric = mean_pred * (1 - mean_pred)
    return aleatoric

def segmentation_epistemic_uncertainty(pred_samples):
    """Compute epistemic uncertainty (model uncertainty)"""
    epistemic = pred_samples.var(dim=0)
    return epistemic

def edl_segmentation_predictions(model, inputs):
    """
    Evidential Deep Learning predictions for segmentation.
    """
    model.eval()
    with torch.no_grad():
        alpha = model(inputs)
        probs = alpha / alpha.sum(dim=1, keepdim=True)
    return probs

def edl_segmentation_uncertainty(alpha):
    """
    Compute uncertainty measures from EDL alpha parameters
    """
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S
    
    # Total uncertainty (entropy of Dirichlet)
    entropy = torch.digamma(S) - torch.sum(probs * torch.digamma(alpha), dim=1, keepdim=True)
    
    # Aleatoric uncertainty (expected entropy)
    aleatoric = torch.sum(probs * (torch.digamma(S + 1) - torch.digamma(alpha + 1)), dim=1, keepdim=True)
    
    # Epistemic uncertainty (mutual information)
    epistemic = entropy - aleatoric
    
    return entropy, aleatoric, epistemic
