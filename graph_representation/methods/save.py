import numpy as np
from graph_representation.MRIData import MRIData


def saveData(data: MRIData):

    np.savez_compressed(
        data.save_graph,
        graph=data.graph,
        patch=data.patch,
        pos=data.pos,
        label=data.label,
    )

    np.savez_compressed(
        data.save_mesh,
        e=data.mesh_edges,
        v=data.mesh_vertices,
        f=data.mesh_faces,
    )

    return
