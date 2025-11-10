import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet, kl_divergence

def evidential_loss(alpha, target, num_classes, epoch, annealing_epochs=20, lambda_reg=0.01):
    # (B, C, H, W)
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S

    # One-hot encode target: (B, H, W) -> (B, C, H, W)
    target_onehot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    # 1. Classification loss: MSE
    mse = ((target_onehot - probs) ** 2).sum(dim=1)

    # 2. Variance term
    var = (probs * (1 - probs) / (S + 1)).sum(dim=1)

    # 3. KL divergence term
    alpha_tilde = alpha * target_onehot + (1 - target_onehot)  # (B, C, H, W)
    beta = torch.ones((1, num_classes, 1, 1), device=alpha.device)

    # Flatten spatial dims for batched Dirichlet ops
    B, C, H, W = alpha_tilde.shape
    alpha_tilde_flat = alpha_tilde.permute(0, 2, 3, 1).reshape(-1, C)
    beta_flat = beta.view(1, C).expand_as(alpha_tilde_flat)

    dirichlet_p = Dirichlet(alpha_tilde_flat)
    dirichlet_q = Dirichlet(beta_flat)

    kl_div_flat = kl_divergence(dirichlet_p, dirichlet_q)  # (B*H*W,)
    kl_div = kl_div_flat.view(B, H, W)

    # 4. Annealing coefficient
    annealing_coef = min(1.0, epoch / annealing_epochs)

    # 5. Total loss (mean over all pixels)
    loss = mse + var + lambda_reg * annealing_coef * kl_div
    return loss.mean()
