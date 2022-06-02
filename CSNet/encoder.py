import timm
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Patch Encoder Layer (pel)
        _cnn = timm.create_model("resnet18", pretrained=False)
        layer0 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), _cnn.bn1, _cnn.act1, _cnn.maxpool)
        self.encoder = nn.Sequential(layer0, _cnn.layer1, _cnn.layer2, _cnn.layer3, _cnn.layer4)
        # Pooling
        self.pooling = nn.AdaptiveMaxPool2d([1, 1])
        # FC
        self.classify = nn.Linear(512, 3)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        x = self.pooling(x).squeeze()
        x = self.classify(x)
        return x
