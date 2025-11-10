import torch
import torch.nn.functional as F

def evidential_loss(alpha, target, num_classes, epoch, annealing_epochs=10, lambda_reg=0.001):
    S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W)
    probs = alpha / S  # (B, C, H, W)

    # One-hot encode target: (B, H, W) -> (B, C, H, W)
    target_onehot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    # 1. Classification loss: MSE
    mse = ((target_onehot - probs) ** 2).sum(dim=1)  # (B, H, W)
    
    # 2. Variance term
    var = (probs * (1 - probs) / (S + 1)).sum(dim=1)  # (B, H, W)
    
    # 3. KL divergence with correct-class masking
    # α̃ = y + (1-y)*α  →  resets correct class to 1, keeps wrong classes as α
    alpha_tilde = target_onehot + (1 - target_onehot) * alpha  # (B, C, H, W)
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)  # (B, 1, H, W)
    
    # KL(Dir(α̃) || Dir(1...1))
    beta = torch.ones((1, num_classes, 1, 1), device=alpha.device)
    
    # Log Beta function terms
    lnB_alpha = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    lnB_beta = torch.lgamma(torch.tensor(float(num_classes), device=alpha.device)) - torch.lgamma(beta).sum()  # scalar
    
    # Digamma terms
    digamma_term = (alpha_tilde - beta) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))  # (B, C, H, W)
    
    # Sum over classes
    kl_div = digamma_term.sum(dim=1, keepdim=True) + lnB_alpha - lnB_beta  # (B, 1, H, W)
    kl_div = kl_div.squeeze(1)  # (B, H, W)

    # Annealing coefficient
    annealing_coef = min(1.0, epoch / annealing_epochs)

    # Total loss: average over all pixels
    loss = mse + var + lambda_reg * annealing_coef * kl_div
    return loss.mean()
