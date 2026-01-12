import pathlib 

from odl.contrib.graphs.graph_interface import create_graph_from_geometry
from odl.contrib.datasets.ct.detect import detect_geometry, detect_ray_trafo, preprocess_sinogram

if __name__ == '__main__':
    
    path_to_data = pathlib.Path(__file__).parent.resolve().joinpath('2detect/slice00001/Mode2')

    geometry = detect_geometry(n_voxels = 512)

    ray_transform = detect_ray_trafo(impl='numpy', device='cpu')

    sinogram = preprocess_sinogram(path_to_data)

    graph = create_graph_from_geometry(geometry, 'GLM', 'torch_geometric')

    print(graph)