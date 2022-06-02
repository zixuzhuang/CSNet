import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.autograd import Variable

MIN_NUM_PATCHES = 16
__all__ = ["ResNet", "resnet10", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet200"]


def get_args(parser):
    parser.add_argument("-f", type=int, default=0)
    parser.add_argument("-c", type=str, default="comparesions/MedT3D/medt_subject.yaml")
    parser.add_argument("-t", type=bool, default=False)
    args = parser.parse_args()
    return args


# def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
#     # 3x3x3 convolution with padding
#     return nn.Conv3d(in_planes, out_planes, kernel_size=3, dilation=dilation, stride=stride, padding=dilation, bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, sample_input_D, sample_input_H, sample_input_W, n_classes, shortcut_type="B", no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        # self.conv_seg = nn.Sequential(
        #                                 nn.ConvTranspose3d(
        #                                 512 * block.expansion,
        #                                 32,
        #                                 2,
        #                                 stride=2
        #                                 ),
        #                                 nn.BatchNorm3d(32),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv3d(
        #                                 32,
        #                                 32,
        #                                 kernel_size=3,
        #                                 stride=(1, 1, 1),
        #                                 padding=(1, 1, 1),
        #                                 bias=False),
        #                                 nn.BatchNorm3d(32),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv3d(
        #                                 32,
        #                                 num_seg_classes,
        #                                 kernel_size=1,
        #                                 stride=(1, 1, 1),
        #                                 bias=False)
        #                                 )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride, no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.conv_seg(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet10(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False,
    )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                            )
                        ),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT3D(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, device, image_size=(20, 448, 448), num_classes=3, pool="cls", channels=1, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        assert all([image_size[d] % patch_size[d] == 0 for d in range(3)])
        self.device = device
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) * (image_size[2] // patch_size[2])
        patch_dim = channels * patch_size[0] * patch_size[1] * patch_size[2]
        assert num_patches > MIN_NUM_PATCHES, f"your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size"
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.patch_size = patch_size

        self.resnet50 = resnet50(
            sample_input_D=patch_size[0],
            sample_input_H=patch_size[1],
            sample_input_W=patch_size[2],
            n_classes=3,
        )
        self.resnetlayer0 = nn.Sequential(
            self.resnet50.conv1,
            self.resnet50.bn1,
            self.resnet50.relu,
            self.resnet50.maxpool,
        )
        self.resnetlayer1 = self.resnet50.layer1
        # self.resnetlayer2 = self.resnet50.layer2
        p0 = math.ceil(patch_size[0] / 4)
        p1 = math.ceil(patch_size[1] / 4)
        self.patch_pooling = nn.AvgPool3d((p0, p1, p1))
        self.patch_to_embedding = nn.Conv3d(256, dim, (p0, p1, p1), (p0, p1, p1))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # self.patch_to_embedding = nn.Conv3d(1, dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.pretrain()

    def forward(self, img, mask=None):
        p = self.patch_size
        x = rearrange(
            img,
            "b c (x p1) (y p2) (z p3) -> b c (x y z) p1 p2 p3",
            p1=p[0],
            p2=p[1],
            p3=p[2],
        )
        b, c, n, p1, p2, p3 = x.shape
        x = rearrange(x, "b c n p1 p2 p3-> (b n) c p1 p2 p3")
        # print(img.shape)
        # print(x.shape)
        x = self.resnetlayer0(x)
        # print(x.shape)
        x = self.resnetlayer1(x)
        x = self.patch_to_embedding(x)
        # x = self.patch_pooling(x)
        # x = x.view(x.shape[:2])
        # print(x.shape)
        x = x.view(b, n, -1)
        # print(x.shape)
        # exit()
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def _forward(self, img, mask=None):
        # print(img.shape)
        x = self.patch_to_embedding(img)
        # print(x.shape)
        x = rearrange(x, "b d n0 n1 n2-> b (n0 n1 n2) d")
        # print(x.shape)
        # exit()
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def pretrain(self):
        _pretrain_dict = torch.load("results/checkpoints/resnet_50_23dataset.pth", map_location=self.device)
        _net_dict = self.state_dict()
        _pretrained_dict = {k: v for k, v in _pretrain_dict.items() if k in _net_dict}
        _net_dict.update(_pretrained_dict)
        self.load_state_dict(_net_dict)


def medT(device):
    return ViT3D(image_size=(24, 512, 512), patch_size=(2, 32, 32), num_classes=3, device=device, dim=768, depth=6, heads=16, mlp_dim=1024, dropout=0.1, emb_dropout=0.1)


if __name__ == "__main__":
    device = "cpu"
    v3d = ViT3D(image_size=(24, 512, 512), patch_size=(4, 32, 32), num_classes=3, device=device, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)
    img3d = torch.randn(2, 1, 24, 512, 512)
    preds3d = v3d(img3d)
