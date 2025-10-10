import torch
import torch.nn.functional as F

def evidential_loss(alpha, target, num_classes, epoch, annealing_epochs=10, lambda_reg=0.001):
    """
    Evidential loss with KL annealing.
    alpha: Dirichlet parameters (batch_size, num_classes)
    target: integer class labels (batch_size,)
    epoch: current training epoch (int)
    annealing_epochs: number of epochs to reach full regularization
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / S

    target_onehot = F.one_hot(target, num_classes=num_classes).float()

    # Expected MSE under Dirichlet
    mse = torch.sum((target_onehot - probs) ** 2, dim=1)
    var = torch.sum(probs * (1 - probs) / (S + 1), dim=1)
    loss_data = mse + var

    # KL divergence (regularizer)
    alpha_hat = (alpha - 1) * (1 - target_onehot) + 1
    kl_div = torch.sum(
        (alpha_hat - 1) * (torch.digamma(alpha_hat) - torch.digamma(torch.sum(alpha_hat, dim=1, keepdim=True))),
        dim=1
    )

    # Annealing coefficient γ
    annealing_coef = min(1.0, epoch / annealing_epochs)

    # Total loss
    loss = loss_data + lambda_reg * annealing_coef * kl_div

    return torch.mean(loss)
