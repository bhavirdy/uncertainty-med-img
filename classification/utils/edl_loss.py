import torch
import torch.nn.functional as F

def evidential_loss(alpha, target, num_classes, epoch, annealing_epochs=10, lambda_reg=0.001):
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S

    # One-hot
    target_onehot = F.one_hot(target, num_classes=num_classes).float()

    # MSE + variance
    diff = target_onehot - probs
    mse = (diff ** 2).sum(dim=1)
    var = (probs * (1 - probs) / (S + 1)).sum(dim=1)
    loss_data = mse + var

    # KL divergence
    K = alpha.size(1)
    beta = torch.ones(1, K, device=alpha.device)
    S_alpha = S
    S_beta = K
    lnB_alpha = torch.lgamma(S_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    lnB_beta = torch.lgamma(torch.tensor(S_beta, device=alpha.device)) - torch.lgamma(beta).sum()
    digamma_diff = torch.digamma(alpha) - torch.digamma(S_alpha)
    kl_div = ((alpha - beta) * digamma_diff).sum(dim=1, keepdim=True) + lnB_alpha - lnB_beta
    kl_div = kl_div.squeeze(1)

    # Annealing
    annealing_coef = torch.clamp(torch.tensor(epoch / annealing_epochs, device=alpha.device), 0.0, 1.0)

    loss = loss_data + lambda_reg * annealing_coef * kl_div
    return loss.mean()
