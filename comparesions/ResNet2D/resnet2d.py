import timm
import torch.nn as nn


def get_args(parser):
    parser.add_argument("-f", type=int, default=0)
    parser.add_argument("-t", type=bool, default=False)
    parser.add_argument("-c", type=str, default="comparesions/ResNet2D/resnet18_patch.yaml")
    args = parser.parse_args()
    return args


class ResNet2D(nn.Module):
    def __init__(self, resnet):
        super().__init__()

        # CNN basic blocks
        _resnet = timm.create_model(resnet, pretrained=True)
        layer0 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), _resnet.bn1, _resnet.act1, _resnet.maxpool)
        self.cnn = nn.Sequential(layer0, _resnet.layer1, _resnet.layer2, _resnet.layer3, _resnet.layer4)
        # Pooling
        self.pooling = nn.AdaptiveMaxPool2d([1, 1])
        # FC
        last_dim = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}
        self.classify = nn.Linear(last_dim[resnet], 3)

    def forward(self, x):
        for layer in self.cnn:
            x = layer(x)
        x = self.pooling(x).squeeze()
        x = self.classify(x)
        return x
