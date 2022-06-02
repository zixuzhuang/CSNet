import argparse

from graph_representation.methods.edge import extractEdges
from graph_representation.methods.fov import adjustFOV
from graph_representation.methods.mesh import makeMesh
from graph_representation.methods.patch import extractPatch
from graph_representation.methods.save import saveData
from graph_representation.methods.vertex import extractVertex
from graph_representation.MRIData import MRIData
from utils.utils_img import show_nii

if __name__ == "__main__":
    # Read input
    parser = argparse.ArgumentParser(description="Graph Representation")
    parser.add_argument("-org", type=str, default="data/MRI/org.nii.gz", help="MRI file")
    parser.add_argument("-seg", type=str, default="data/MRI/seg.nii.gz", help="segmentation file")
    parser.add_argument("-grade", type=int, default=0, help="cartilage defect grade, default: 0")
    parser.add_argument("-graph", type=str, default="data/knee_graph/example_0.npz", help="save path of knee graph")
    parser.add_argument("-mesh", type=str, default="data/knee_mesh/example_0.npz", help="save path of knee mesh for visualization")
    args = parser.parse_args()

    # Process data
    data = MRIData(args)
    adjustFOV(data)
    extractVertex(data)
    extractEdges(data)
    extractPatch(data)
    makeMesh(data)
    saveData(data)
