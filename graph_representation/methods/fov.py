import skimage.feature
import skimage.measure
import skimage.morphology
import torch
from graph_representation.MRIData import MRIData


def adjustFOV(data: MRIData):

    # Height limitation
    petalla = data.bone[2]
    h_min = max(petalla.nonzero()[:, 1].min().item() - 30, 0)
    h_max = h_min + 380

    # Width limitation
    w_min = max(petalla.nonzero()[:, 2].min().item() + 10, 0)
    w_max = w_min + 380

    # Extract surface
    for b in range(data.num):
        for s in range(data.slice):
            slice = data.bone[b, s].clone().numpy()
            for _ in range(3):
                slice = skimage.morphology.binary_dilation(slice)
            slice = skimage.morphology.binary_dilation(slice).astype(int) - slice.astype(int)
            data.surface[b, s] = torch.tensor(slice)

    # Update bone segmentation FOV
    data.surface[..., :h_min, :] = 0
    data.surface[..., h_max:, :] = 0
    data.surface[..., :, :w_min] = 0
    data.surface[..., :, w_max:] = 0

    return data
