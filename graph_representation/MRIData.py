from os.path import join

import numpy as np
import SimpleITK as sitk
import torch
from SimpleITK import GetArrayFromImage as GAFI
# from torch.nn.functional import interpolate, pad
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode as im
from utils.utils_img import normalize, to_binary_mask

import graph_representation.config as cfg

# from utils.morphology import show_slices, split_les, split_seg


class MRIData(object):
    def __init__(self, args) -> None:
        super().__init__()

        # Read data
        mri_data = sitk.DICOMOrient(sitk.ReadImage(args.org), "PIL")
        seg_data = sitk.DICOMOrient(sitk.ReadImage(args.seg), "PIL")
        mri_img = torch.tensor(GAFI(mri_data).astype(np.float32), dtype=torch.float32)
        seg_img = torch.tensor(GAFI(seg_data).astype(np.int32), dtype=torch.int32)

        # Normalize spacing
        spacing = list(mri_data.GetSpacing())
        scale = spacing[0] / cfg.STD_SPACING
        size = (seg_img.shape[0], int(seg_img.shape[1] * scale), int(seg_img.shape[2] * scale))
        mri_img = Resize(size=size[1:], interpolation=im.BILINEAR)(mri_img)
        seg_img = Resize(size=size[1:], interpolation=im.NEAREST)(seg_img)

        # Normalize intensity
        self.mri_img = normalize(mri_img)

        # Extract bone segmentation
        self.bone = to_binary_mask(seg_img, num_cls=6)[np.array(cfg.bones_idx) - 1]

        # Add infomation
        self.shape = mri_img.shape[1:]
        self.slice = mri_img.shape[0]
        self.space = [spacing[2], cfg.STD_SPACING, cfg.STD_SPACING]
        self.num = 3

        # Other parematers
        self.surface = torch.zeros(self.bone.shape, dtype=torch.int32)
        self.thick = np.round(self.space[0] / self.space[2], 3)
        self.T_dist = (self.thick ** 2 + cfg.PATCH_SIZE ** 2) ** 0.5 * 0.8
        self.v_2d = []
        self.v_3d = []
        self.v_idx = []
        self.edges = None
        self.patch = None
        self.pos = None
        self.graph = None
        self.mask = None
        self.mesh_edges = None
        self.mesh_vertices = None
        self.mesh_faces = None

        # Save path
        self.label = torch.tensor(args.grade, dtype=torch.long)
        self.save_graph = args.graph
        self.save_mesh = args.mesh
        return
