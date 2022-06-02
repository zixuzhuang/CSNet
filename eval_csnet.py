import argparse

import numpy as np
import torch

from CSNet.csnet import CSNet
from CSNet.dataloader import CSNetData
from CSNet.encoder import Encoder
from utils.graphcam import calGraphCAM
from utils.visualization import plotAttentionMap

if __name__ == "__main__":
    # Read parameters
    parser = argparse.ArgumentParser(description="Evaluation CSNet")
    parser.add_argument("-graph", type=str, default="data/knee_graph/example_0.npz")
    parser.add_argument("-mesh", type=str, default="data/knee_mesh/example_0.npz")
    parser.add_argument("-net", type=str, default="None")
    parser.add_argument("-img", type=str, default="data/visualization/example_0.png")
    args = parser.parse_args()

    # Load knee graph and knee mesh
    data = CSNetData()
    graph_data = np.load(args.graph, allow_pickle=True)
    data.patch = torch.tensor(graph_data["patch"], dtype=torch.float32)[:, None, :, :]
    data.graph = graph_data["graph"].item()
    data.label = torch.tensor(graph_data["label"], dtype=torch.long)
    data.pos = torch.tensor(graph_data["pos"], dtype=torch.float32)
    data.to("cuda")
    mesh_data = np.load(args.mesh, allow_pickle=True)

    # Load CSNet
    if args.net == "None":
        encoders = Encoder()
        net = CSNet(encoders)
    else:
        net = torch.load(args.net)
    net = net.to("cuda")
    net.eval()
    torch.set_grad_enabled(False)

    # Calculate graph attention map and show it in the knee mesh
    pred, feat = net(data)
    grade = torch.argmax(pred, dim=1)
    cam = calGraphCAM(feat, grade, net.gclassifier)
    plotAttentionMap(grade, cam, mesh_data, args.img)
