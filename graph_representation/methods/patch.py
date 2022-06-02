import dgl
import graph_representation.config as cfg
import torch
from graph_representation.MRIData import MRIData
from torchvision.transforms import CenterCrop, Pad


def paintingPatch(slice, vertex):
    p = cfg.PATCH_SIZE
    p1, p2 = p - p // 2, p + p // 2
    linewidth = 3
    shape = slice.shape
    slice = Pad([p], fill=0.0)(slice)
    mask = torch.zeros(slice.shape, dtype=bool)
    for xy in vertex:
        x, y = xy
        x, y = int(x), int(y)
        mask[x + p1 : x + p2, y + p1 : y + p2] = 1
        # for _i in range(linewidth):
        #     mask[x + p1 + _i, y + p1 : y + p2] = 1
        #     mask[x + p2 - _i, y + p1 : y + p2] = 1
        #     mask[x + p1 : x + p2, y + p1 + _i] = 1
        #     mask[x + p1 : x + p2, y + p2 - _i] = 1
    mask = CenterCrop(shape)(mask)
    return mask


def extract_patch(slice, vertex):
    p = cfg.PATCH_SIZE
    p1, p2 = p - p // 2, p + p // 2
    slice = Pad([p], fill=0)(slice)
    patch = []
    for xy in vertex:
        x, y = xy
        x, y = int(x), int(y)
        patch.append(slice[x + p1 : x + p2, y + p1 : y + p2])
    return torch.stack(patch)


def extractPatch(data: MRIData):
    ptaches = []
    mask = torch.zeros(data.mri_img.shape, dtype=bool)

    for b in range(data.num):
        for s in range(data.slice):
            if len(data.v_2d[b][s]) > 0:
                _ptchs = extract_patch(data.mri_img[s], data.v_2d[b][s])
                ptaches.append(_ptchs)
                mask[s] += paintingPatch(data.mri_img[s], data.v_2d[b][s])
    data.patch = torch.cat(ptaches, dim=0).type(torch.float32)

    data.pos = torch.tensor(data.v_3d, dtype=torch.float32)
    data.pos[:, 0] = data.pos[:, 0] / ((data.slice - 1) * data.thick)
    data.pos[:, 1] = data.pos[:, 1] / data.shape[0]
    data.pos[:, 2] = data.pos[:, 2] / data.shape[1]
    graph = dgl.graph((data.edges[0], data.edges[1]))
    data.graph = dgl.add_self_loop(graph)
    data.mask = mask
    return data
