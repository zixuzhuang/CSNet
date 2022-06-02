import timm
import torch
import torch.nn as nn
from torch.nn import functional as F


def get_args(parser):
    parser.add_argument("-f", type=int, default=0)
    parser.add_argument("-t", type=bool, default=True)
    parser.add_argument("-c", type=str, default="comparesions/DC_MT/dcmt_slice.yaml")
    args = parser.parse_args()
    return args


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN basic blocks
        _cnn = timm.create_model("resnet18", pretrained=True)
        layer0 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), _cnn.bn1, _cnn.act1, _cnn.maxpool)
        self.cnn = nn.Sequential(layer0, _cnn.layer1, _cnn.layer2, _cnn.layer3, _cnn.layer4)

        # pooling
        self.pooling = nn.AdaptiveMaxPool2d([1, 1])

        # FC
        self.classify = nn.Linear(512, 3)

    def forward(self, x):
        for layer in self.cnn:
            x = layer(x)
        fm = x
        x = self.pooling(x).squeeze()
        x = self.classify(x)
        return x, fm


class MeanTeacherNet(nn.Module):
    def __init__(self, lamb=0.9):
        super().__init__()

        self.lamba = lamb

        self.student = BaseNet()
        self.teacher = BaseNet()

        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def ema_update(self):
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = param_t.data * self.lamba + param_s.data * (1.0 - self.lamba)


def cls_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = torch.softmax(input_logits, dim=1)
    target_softmax = torch.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    bs = input_logits.size()[0]
    return F.mse_loss(input_softmax, target_softmax, reduction="sum") / (num_classes * bs)


def att_mse_loss(mask, cams):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert mask.size() == cams.size() and len(mask.size()) == 4
    mse_loss = F.mse_loss(mask, cams, reduction="none").sum((2, 3))
    norm = (mask.sum((2, 3)) + cams.sum((2, 3))).sum()
    mse_loss = torch.sum(mse_loss) / torch.clamp(norm, min=1e-5)
    return mse_loss


def normalization(feautures):
    B, _, H, W = feautures.size()
    outs = feautures.squeeze(1)
    outs = outs.view(B, -1)
    outs_min = outs.min(dim=1, keepdim=True)[0]
    outs_max = outs.max(dim=1, keepdim=True)[0]
    norm = outs_max - outs_min
    norm[norm == 0] = 1e-5
    outs = (outs - outs_min) / norm
    outs = outs.view(B, 1, H, W)
    return outs


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FABlock(nn.Module):
    def __init__(self, in_channels, norm_layer=None, reduction=8):
        super(FABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(in_channels, 1)
        self.channel_fc = nn.Sequential(nn.Linear(in_channels, in_channels // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(in_channels // reduction, in_channels, bias=False))
        self.conv2 = conv1x1(in_channels, in_channels)

        self.conv3 = conv1x1(in_channels, 1)
        self.conv4 = conv3x3(1, 1)
        self.bn4 = norm_layer(1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        # channel attention
        y = self.conv1(x).view(B, 1, -1)
        y = F.softmax(y, dim=-1)
        y = y.permute(0, 2, 1).contiguous()
        y = torch.matmul(x.view(B, C, -1), y).view(B, -1)
        y = self.channel_fc(y)
        y = torch.sigmoid(y).unsqueeze(2).unsqueeze(3).expand_as(x)

        x_y = self.conv2(x)
        x_y = x_y * y

        # position attention
        x_y_z = self.conv3(x_y)
        z = self.conv4(x_y_z)
        z = self.bn4(z)
        z = torch.sigmoid(z)
        x_y_z = x_y_z * z

        out = self.gamma * x_y_z + x
        attention_outs = normalization(self.gamma * x_y_z)

        return out, attention_outs
