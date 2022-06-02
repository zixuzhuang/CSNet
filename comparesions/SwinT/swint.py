import torch
from swin_transformer_pytorch import SwinTransformer


def swint_slice():
    return SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), channels=1, num_classes=3, head_dim=32, window_size=8, downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True)


def swint_patch():
    return SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), channels=1, num_classes=3, head_dim=32, window_size=4, downscaling_factors=(2, 2, 2, 2), relative_pos_embedding=True)


def get_args(parser):
    parser.add_argument("-f", type=int, default=0)
    parser.add_argument("-c", type=str, default="comparesions/SwinT/swint_patch.yaml")
    parser.add_argument("-t", type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # x = torch.randn(64, 1, 64, 64)
    x = torch.randn(2, 1, 256, 256)
    # net = UNet(3, 2, (64, 64*2, 64*4, 64*8, 64*16)).cuda()
    net = swint_slice()
    y = net(x)
    print(y.shape)
    # cls = Classifier()
