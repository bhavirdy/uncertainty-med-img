import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, encoder_name='resnet34', encoder_weights='imagenet', dropout_p=0.0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # SMP UNet with ResNet34 encoder and ImageNet weights
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=n_classes
        )

    def forward(self, x):
        return self.model(x)

class UNetEDL(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=False, dropout_p=0.5):
        super(UNetEDL, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Use SMP UNet with 2-class output as backbone for EDL
        self.backbone = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=n_channels,
            classes=n_classes
        )
        
        # Evidence head: ensure non-negative evidence per class
        self.evidence_head = nn.Sequential(
            nn.Softplus()
        )

    def forward(self, x):
        logits = self.backbone(x)  # [B, 2, H, W]
        evidence = self.evidence_head(logits)
        alpha = evidence + 1
        return alpha
