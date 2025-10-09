import torch

def evidential_loss(alpha, target, num_classes, lambda_reg=0.001):
    """
    alpha: Dirichlet parameters from model (batch_size, num_classes)
    target: ground truth labels (batch_size)
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / S

    target_onehot = torch.nn.functional.one_hot(target, num_classes=num_classes).float()

    loglikelihood = torch.sum(target_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    evidence_regularizer = lambda_reg * torch.sum((1 - target_onehot) * alpha, dim=1)
    loss = loglikelihood + evidence_regularizer
    return torch.mean(loss)
