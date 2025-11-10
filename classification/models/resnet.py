import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.resnet import Bottleneck

class ResNet50Deterministic(nn.Module):
    def __init__(self, num_classes, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        
        # Load pretrained ResNet50
        self.model = resnet50(weights=weights)

        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace FC with trainable classifier
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Make FC trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

class ResNet50MCDO(nn.Module):
    def __init__(self, num_classes, weights=ResNet50_Weights.DEFAULT, dropout_p=0.3):
        super().__init__()
        self.model = resnet50(weights=weights)
        
        # Freeze pre-trained backbone
        for p in self.model.parameters():
            p.requires_grad = False

        # Inject Dropout2d into Bottleneck blocks
        for layer in [self.model.layer2, self.model.layer3, self.model.layer4]:
            for block in layer:
                # Add dropout after each residual block's ReLU
                block.dropout = nn.Dropout2d(p=dropout_p)
                
                # Modify forward pass of the block to include dropout
                orig_forward = block.forward
                def new_forward(x, orig_forward=orig_forward, dropout=block.dropout):
                    out = orig_forward(x)
                    return dropout(out)
                block.forward = new_forward

        # Replace final fully connected layer with dropout + linear
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class ResNet50EDL(nn.Module):
    def __init__(self, num_classes, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        
        # Load pretrained ResNet50
        self.model = resnet50(weights=weights)
        
        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, num_classes),
            nn.Softplus() # ensures non-negative evidence outputs
        )
        
        # Make FC trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        evidence = self.model(x)
        alpha = evidence + 1  # Dirichlet parameters (alpha >= 1)
        return alpha
