from graph_representation.MRIData import MRIData
from utils.utils_img import show_seg, show_slices


def test(data: MRIData):
    show_seg(data.mri_img, data.mask, "./mask.png", 4, 0.6, 5)

