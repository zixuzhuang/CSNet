import torch
from vit_pytorch import ViT


def vit_slice():
    return ViT(image_size=256, patch_size=32, channels=1, num_classes=3, dim=768, depth=12, heads=16, mlp_dim=1024, dropout=0.1, emb_dropout=0.1)


def vit_patch():
    return ViT(image_size=64, patch_size=8, channels=1, num_classes=3, dim=768, depth=12, heads=16, mlp_dim=1024, dropout=0.1, emb_dropout=0.1)


def get_args(parser):
    parser.add_argument("-f", type=int, default=0)
    parser.add_argument("-c", type=str, default="comparesions/ViT/vit_patch.yaml")
    parser.add_argument("-t", type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    x = torch.randn(64, 1, 256, 256)
    # net = UNet(3, 2, (64, 64*2, 64*4, 64*8, 64*16)).cuda()
    net = vit_slice()
    y = net(x)
    print(y.shape)
    # cls = Classifier()
