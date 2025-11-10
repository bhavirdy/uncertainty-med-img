import torch
import torch.nn as nn

def mcdo_predictions(model, inputs, n_samples=50):
    # Enable dropout
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = torch.softmax(model(inputs), dim=1)
            preds.append(out.unsqueeze(0))
    return torch.cat(preds, dim=0)

def predictive_mean(pred_samples):
    return pred_samples.mean(dim=0)
