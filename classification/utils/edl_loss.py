import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet, kl_divergence

def evidential_loss(alpha, target, num_classes, epoch, annealing_epochs=10, lambda_reg=0.01):
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S

    # One-hot encode target
    target_onehot = F.one_hot(target, num_classes=num_classes).float()

    # 1. Classification loss (MSE)
    mse = ((target_onehot - probs) ** 2).sum(dim=1)

    # 2. Variance term
    var = (probs * (1 - probs) / (S + 1)).sum(dim=1)

    # 3. KL divergence term
    alpha_tilde = target_onehot + (1 - target_onehot) * alpha
    beta = torch.ones_like(alpha)

    dirichlet_p = Dirichlet(alpha_tilde)
    dirichlet_q = Dirichlet(beta)

    kl_div = kl_divergence(dirichlet_p, dirichlet_q)

    # 4. Annealing coefficient
    annealing_coef = min(1.0, epoch / annealing_epochs)

    # 5. Total loss
    loss = mse + var + lambda_reg * annealing_coef * kl_div
    return loss.mean()
