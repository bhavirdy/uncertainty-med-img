import torch
import torch.nn.functional as F

def evidential_loss(alpha, target, num_classes, epoch, annealing_epochs=10, lambda_reg=0.001):
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S
    target_onehot = F.one_hot(target, num_classes=num_classes).float()

    # Data term
    mse = (target_onehot - probs).pow(2).sum(dim=1)
    var = (probs * (1 - probs) / (S + 1)).sum(dim=1)
    loss_data = mse + var

    # Precompute constants
    K = alpha.size(1)
    beta = torch.ones((1, K), device=alpha.device)
    S_alpha = S
    lnB_alpha = torch.lgamma(S_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    lnB_beta = torch.lgamma(K) - K * torch.lgamma(torch.tensor(1., device=alpha.device))

    # KL term
    digamma_diff = torch.digamma(alpha) - torch.digamma(S_alpha)
    kl_div = ((alpha - 1) * digamma_diff).sum(dim=1) + (lnB_alpha.squeeze(1) - lnB_beta)

    # Annealing
    annealing_coef = min(1.0, epoch / annealing_epochs)
    loss = loss_data + lambda_reg * annealing_coef * kl_div
    return loss.mean()
