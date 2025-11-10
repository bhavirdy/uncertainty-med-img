import torch
import torch.nn.functional as F

def evidential_loss(alpha, target, num_classes, epoch, annealing_epochs=10, lambda_reg=0.001):
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S

    # One-hot encode target
    target_onehot = F.one_hot(target, num_classes=num_classes).float()

    # 1. Classification loss: MSE
    mse = ((target_onehot - probs) ** 2).sum(dim=1)
    
    # 2. Variance term
    var = (probs * (1 - probs) / (S + 1)).sum(dim=1)
    
    # 3. KL divergence with masking
    # α̃ = y + (1-y)*α  →  resets correct class to 1, keeps wrong classes as α
    alpha_tilde = target_onehot + (1 - target_onehot) * alpha
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    
    # KL(Dir(α̃) || Dir(1...1))
    beta = torch.ones_like(alpha)
    
    lnB_alpha = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
    lnB_beta = torch.lgamma(torch.tensor(float(num_classes), device=alpha.device)) - torch.lgamma(beta).sum(dim=1, keepdim=True)
    
    digamma_term = (alpha_tilde - beta) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))
    kl_div = (digamma_term.sum(dim=1, keepdim=True) + lnB_alpha - lnB_beta).squeeze(1)

    # Annealing coefficient
    annealing_coef = min(1.0, epoch / annealing_epochs)

    # Total loss
    loss = mse + var + lambda_reg * annealing_coef * kl_div
    return loss.mean()
