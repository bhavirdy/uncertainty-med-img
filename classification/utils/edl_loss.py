import torch
import torch.nn.functional as F

def evidential_loss(alpha, target, num_classes, epoch, annealing_epochs=10, lambda_reg=0.01):
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / S
    target_onehot = F.one_hot(target, num_classes=num_classes).float()

    # Data (MSE) term
    mse = torch.sum((target_onehot - probs) ** 2, dim=1)
    var = torch.sum(probs * (1 - probs) / (S + 1), dim=1)
    loss_data = mse + var

    # KL divergence
    def KL(alpha):
        K = alpha.shape[1]
        beta = torch.ones((1, K), device=alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_beta = torch.lgamma(S_beta) - torch.sum(torch.lgamma(beta), dim=1, keepdim=True)
        digamma_diff = torch.digamma(alpha) - torch.digamma(S_alpha)
        kl = torch.sum((alpha - beta) * digamma_diff, dim=1, keepdim=True) + lnB_alpha - lnB_beta
        return kl.squeeze(1)

    kl_div = KL(alpha)

    # Annealing
    annealing_coef = min(1.0, epoch / annealing_epochs)

    loss = loss_data + lambda_reg * annealing_coef * kl_div
    return torch.mean(loss)
