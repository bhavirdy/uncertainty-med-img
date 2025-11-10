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

class ResNet50MCDO_og(nn.Module):
    def __init__(self, num_classes, weights=ResNet50_Weights.DEFAULT, dropout_p=0.5):
        super().__init__()

        # Load pretrained ResNet50
        self.model = resnet50(weights=weights)

        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Wrap each ResNet stage with Dropout
        self.model.layer1 = nn.Sequential(self.model.layer1, nn.Dropout2d(p=dropout_p))
        self.model.layer2 = nn.Sequential(self.model.layer2, nn.Dropout2d(p=dropout_p))
        self.model.layer3 = nn.Sequential(self.model.layer3, nn.Dropout2d(p=dropout_p))
        self.model.layer4 = nn.Sequential(self.model.layer4, nn.Dropout2d(p=dropout_p))

        # Replace FC with Dropout + Linear
        self.model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(self.model.fc.in_features, num_classes))

        # Make FC trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

class BottleneckWithDropout(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 dropout_p=0.5):
        super().__init__(
            inplanes, planes, stride=stride, downsample=downsample,
            groups=groups, base_width=base_width, dilation=dilation,
            norm_layer=norm_layer
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)  # MC Dropout here

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)  # optional, can remove if needed

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# --- 2. ResNet50 MC Dropout ---
class ResNet50MCDO(nn.Module):
    def __init__(self, num_classes, weights=ResNet50_Weights.DEFAULT, dropout_p=0.3):
        super().__init__()
        self.model = resnet50(weights=weights)

        # Freeze backbone initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace Bottleneck blocks in layers 2–4 with dropout version
        for layer_name in ['layer2', 'layer3', 'layer4']:
            old_layer = getattr(self.model, layer_name)
            new_blocks = []
            for b in old_layer:
                if isinstance(b, Bottleneck):
                    new_blocks.append(
                        BottleneckWithDropout(
                            inplanes=b.conv1.in_channels,
                            planes=b.conv2.out_channels,
                            stride=b.conv2.stride[0],
                            downsample=b.downsample,
                            dropout_p=dropout_p
                        )
                    )
            setattr(self.model, layer_name, nn.Sequential(*new_blocks))

        # Replace FC with Dropout + Linear
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

        # Make FC trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

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
