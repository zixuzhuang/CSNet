import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import torch


def plotAttentionMap(grade, cam, mesh_data, save_path):
    # Load mesh
    edge_rough = torch.tensor(mesh_data["e"], dtype=torch.long)
    vertex_rough = torch.tensor(mesh_data["v"], dtype=torch.long)
    face_rough = torch.tensor(mesh_data["f"], dtype=torch.long)
    x_rough, z_rough, y_rough = np.array(vertex_rough, dtype=np.float64).transpose()
    z_rough = -z_rough

    # No subdivided surfaces in this demo
    vertex_refine = vertex_rough
    edge_refine = edge_rough
    face_refine = face_rough
    x_refine = x_rough
    y_refine = y_rough
    z_refine = z_rough

    # # Better visualization with more subdivided surfaces
    # in_path_refine = join("data/P16_Z1_V2_MESH/mesh", data.name[0])
    # mesh_refine = np.load(in_path_refine)
    # vertex_refine = mesh_refine["v"]
    # edge_refine = mesh_refine["e"]
    # face_refine = mesh_refine["f"]
    # x_refine, z_refine, y_refine = np.array(vertex_refine, dtype=np.float64).transpose()
    # z_refine = -z_refine

    # Registration
    cam_refine = np.zeros([vertex_refine.shape[0], 1])
    cam_tree = scipy.spatial.cKDTree(vertex_rough)
    kd_dist, kd_idx = cam_tree.query(vertex_refine)
    cam_refine = cam[kd_idx]

    # Calculate face color
    f_intensity = []
    for i in range(face_refine.shape[0]):
        _ = np.mean([cam_refine[face_refine[i][0]], cam_refine[face_refine[i][1]], cam_refine[face_refine[i][2]]]) * 255
        f_intensity.append(_)
    f_intensity = np.round(f_intensity, 2).astype(int)

    # Select color for different grade
    if grade == 0:
        cmap = plt.get_cmap("Greys")  # Greys for Grade 0
    elif grade == 1:
        cmap = plt.get_cmap("Greens")  # Greens for Grade 1
    elif grade == 2:
        cmap = plt.get_cmap("OrRd")  # Red for Grade 2

    cNorm = colors.Normalize(vmin=0, vmax=2)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    f_color = scalarMap.to_rgba(f_intensity)
    f_color[:, 3] *= 0.5

    # Create figure
    fig = plt.figure(figsize=(12, 8), facecolor="white", dpi=150)
    ax = plt.axes(fc="whitesmoke", projection="3d")
    ax.set_zlim([-128 * 3, -128])
    plt.xlim([128, 128 * 3])
    plt.ylim([128, 128 * 3])
    ax.set_facecolor("white")
    ax.axis("off")

    # Plot mesh
    T = np.array(face_refine)
    for i in range(len(face_refine)):
        ax.plot_trisurf(x_refine, y_refine, z_refine, triangles=[T[i]], edgecolor=[[0.6, 0.6, 0.6]], color=f_color[i], linewidth=0.1, alpha=0.5, shade=False)
    ax.view_init(elev=0, azim=0)  # 仰角  # 方位角

    # Show or save image
    # plt.show()
    plt.savefig(save_path, bbox_inches="tight")
