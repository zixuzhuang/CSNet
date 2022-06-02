import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax
from einops import rearrange

from CSNet.encoder import Encoder


def get_args(parser):
    parser.add_argument("-f", type=int, default=0)
    parser.add_argument("-g", type=str, default="0")
    parser.add_argument("-t", type=bool, default=False)
    parser.add_argument("-c", type=str, default="configs/csn_subject.yaml")
    parser.add_argument("-net", type=str, default="")
    args = parser.parse_args()
    return args


def get_stage_size(net, ps):
    x = torch.ones([4, 1, ps, ps], dtype=torch.float32).to(next(net.parameters()).device)
    dims_list, size_list = [x.shape[1]], [x.shape[2]]
    with torch.no_grad():
        for layer in net:
            x = layer(x)
            dims_list.append(x.shape[1])
            size_list.append(x.shape[2])
    return dims_list, size_list


class PreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm_out = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm_out(x)
        x = self.net(x)
        return x


class GraphAttention(nn.Module):
    def __init__(self, g_dim, head_n, head_d, dropout):
        super().__init__()
        # Tranformer
        inner_dim = head_d * head_n
        self.head_num = head_n
        self.scale = g_dim ** -0.5
        self.to_qkv = nn.Linear(g_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, g_dim), nn.Dropout(dropout))
        # PreNorm
        self.norm_in = nn.LayerNorm(g_dim)

    def forward(self, s, g):
        s = s[None, ...]
        s = self.norm_in(s)
        with g.local_scope():
            qkv = self.to_qkv(s).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b n) h d", h=self.head_num), qkv)
            # b=1, n=num_patches, h=head_num, d=head_dim

            g.srcdata.update({"q": q, "v": v})
            g.dstdata.update({"k": k})

            # compute edge attention.
            g.apply_edges(fn.u_mul_v("q", "k", "att"))
            g.edata["att"] = g.edata["att"] * self.scale
            g.edata["att"] = edge_softmax(g, g.edata["att"])

            # message passing
            g.update_all(fn.u_mul_e("v", "att", "m"), fn.sum("m", "rst"))
            rst = g.dstdata["rst"]

            # recovery shape
            rst = rearrange(rst, "n h d -> n (h d)")[None, ...]
            rst = self.to_out(rst)[0]
        return rst


class GraphTransformer(nn.Module):
    def __init__(self, g_dim, x_ps, x_dim, head_n, head_d, dropout=0.1):
        super().__init__()
        # x proj
        self.xproj = nn.Conv2d(in_channels=x_dim, out_channels=g_dim, kernel_size=x_ps)

        # Tranformer
        self.norm = nn.LayerNorm(g_dim)
        self.attention = GraphAttention(g_dim, head_n, head_d, dropout)
        self.ff = FeedForward(g_dim, hidden_dim=512, dropout=dropout)

    def forward(self, x, h, g):
        # linear projection
        x = self.xproj(x).squeeze()
        # gcn block
        h = self.norm(h)
        h = self.ff(self.attention(h, g)) + h
        # merge
        h = h + x
        return h


class CSNet(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()

        # Patch Encoder Layer (pel)
        self.encoder = nn.Sequential(
            *encoder.encoder,
        )

        # Hyper Parameter
        head_n = (4, 4, 8, 8)
        head_d = 128
        g_dim = 512
        x_dim, x_ps = get_stage_size(self.encoder, 64)

        # Graph Transformer Layer (gtl)
        self.gtl = nn.ModuleList()
        for i in range(1, 5):
            self.gtl.append(GraphTransformer(g_dim=g_dim, x_ps=x_ps[i + 1], x_dim=x_dim[i + 1], head_n=head_n[i - 1], head_d=head_d))

        # Pooling
        self.pooling = nn.AdaptiveMaxPool2d([1, 1])

        # Position embedding
        self.pe = nn.Linear(3, g_dim)

        # Merge
        self.xproj = nn.Linear(x_dim[-1], g_dim)
        self.graphconv = dglnn.GraphConv(g_dim * 2, g_dim * 2)
        self.gclassifier = nn.Linear(g_dim * 2, 3)

    def forward(self, data):
        x, p, g = data.patch, data.pos, data.graph
        x = self.encoder[0](x)
        h = self.pe(p)
        for i in range(1, 5):
            x = self.encoder[i](x)
            h = self.gtl[i - 1](x, h, g)
        x = self.pooling(x).squeeze()
        x = self.xproj(x)
        x = torch.cat([x, h], dim=1)
        x = self.graphconv(g, x)
        with g.local_scope():
            g.ndata["x"] = x
            mx = dgl.mean_nodes(g, "x")
        gx = self.gclassifier(mx)
        return gx, x
