import sys 
import pathlib
import yaml 

import torch
import numpy as np
import matplotlib.pyplot as plt

from odl.contrib.torch.operator import OperatorModule
from odl.contrib.graphs.graph_interface import create_graph_from_geometry
from odl.contrib.datasets.ct.detect import detect_geometry, detect_ray_trafo, preprocess_sinogram

from glm.utils import plot_image

def main():
    if len(sys.argv) != 1:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython run_demo.py \n")
        sys.exit(1)

    parameters = yaml.safe_load(open("params.yaml"))
    processed_path = pathlib.Path(parameters['demo']['data']['processed_path'])

    ray_transform = detect_ray_trafo(impl='numpy', device='cpu')

    graph = create_graph_from_geometry(
        geometry = detect_geometry(n_voxels = 512), 
        scheme   = 'GLM', 
        backend  = 'torch_geometric')

    sinogram = np.load(processed_path.joinpath('slice00001/mode2/preprocessed_sinogram.npy'))
    target = np.load(processed_path.joinpath('slice00001/mode2/preprocessed_reconstruction.npy'))

    ### We create the backprojection operator
    backprojection = OperatorModule(ray_transform.adjoint)

    ### We load the sinogram to PyTorch and make it the right shape 
    sinogram = torch.from_numpy(sinogram).unsqueeze(0).unsqueeze(0)
    reconstruction = backprojection(sinogram)

    plot_image(
        reconstruction, 
        'src/glm/images/demo/reconstruction', 
        title='Reconstruction'
        )
    
    plot_image(
        sinogram, 
        'src/glm/images/demo/sinogram', 
        title='Sinogram'
        )
    
    plot_image(
        target, 
        'src/glm/images/demo/target', 
        title='Target'
        )

if __name__ == '__main__':
    main()