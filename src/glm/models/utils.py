from typing import Dict,List

import torch
from torch_geometric.nn import Sequential

from odl.contrib.torch.operator import OperatorModule
from odl.applications.tomo.geometry import Geometry
from odl.contrib.graphs.graph_interface import create_graph_from_geometry
from odl.contrib.datasets.ct.detect import detect_geometry, detect_ray_trafo
from torch.nn.parallel import DistributedDataParallel as DDP

from .cnn import ImageCNN, CNN_Module
from .gnn import GLM_Module

class PSNR:
    def __init__(self, reduction = 'mean'):
        self.loss = torch.nn.MSELoss(reduction=reduction)

    def __call__(self, inferred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 20 * torch.log10(target.max() - target.min()) - 10 * torch.log10(
            self.loss(inferred, target)
        )

def load_model(model_name:str, model_parameters:Dict):
    if model_name == 'sinogram_CNN':
        n_channels = model_parameters['n_channels']
        model = torch.nn.Sequential(
            CNN_Module(1, n_channels, model_parameters),
            torch.nn.ReLU(inplace=True),
            CNN_Module(n_channels, n_channels, model_parameters),
            torch.nn.ReLU(inplace=True),
            CNN_Module(n_channels, 1, model_parameters),
            torch.nn.ReLU(inplace=True),
        )

    elif model_name == 'image_CNN':
        model = ImageCNN(model_parameters)
    
    elif model_name == 'GLM':
        n_channels = model_parameters['n_channels']
        model = Sequential('x, edge_index, edge_weight', [
                (
                    GLM_Module(1, n_channels, model_parameters), 
                    'x, edge_index, edge_weight -> x'
                ),
            torch.nn.ReLU(inplace=True),
                (
                    GLM_Module(n_channels, n_channels, model_parameters), 
                    'x, edge_index, edge_weight -> x'
                ),
            torch.nn.ReLU(inplace=True),
                (
                    GLM_Module(n_channels, 1, model_parameters), 
                    'x, edge_index, edge_weight -> x'
                ),
            torch.nn.ReLU(inplace=True),
        ])
    
    else:    
        raise ValueError(f'The model {model_name} does not have an implementation.')
    
    return model

def load_geometry(angles_indices:List=None):
    return detect_geometry(n_voxels = 512, angles_indices=angles_indices)

def load_graph(model_name, geometry):
    if model_name == 'GLM':        
        return create_graph_from_geometry(
            geometry = geometry, 
            scheme   = 'GLM', 
            backend  = 'torch_geometric'
            )
    else:
        return None
    
def set_data_shape(
        model : torch.nn.Sequential | Sequential, 
        batch_size :int, 
        angles_indices : None | List,
        n_measurements : int,          
        tensor, 
        target,
        n_pixels : int = 956, 
        ):
    if isinstance(model, torch.nn.Sequential) or isinstance(model, Sequential):
        return model[0].set_data_shape(
            batch_size, angles_indices, n_measurements, n_pixels, tensor, target
            )
    elif isinstance(model, DDP):
        return set_data_shape(
            model.module, 
            batch_size=batch_size,
            angles_indices = angles_indices,
            n_measurements = n_measurements,          
            tensor = tensor, 
            target = target,
            n_pixels = n_pixels
            )
    else:
        return model.set_data_shape(
            batch_size, angles_indices, n_measurements, n_pixels, tensor, target
            )
    
def get_angles_list_from_downsampling(downsampling : int = 1):
    assert isinstance(downsampling, int)
    if downsampling==1:
        return None
    return [i for i in range(0, 3600, downsampling)]

def load_pseudo_inverse(
        name:str, 
        parameters:Dict, 
        geometry:Geometry,
        device,
        ):
    if name == 'backprojection':
        n_voxels = parameters['n_voxels']
        impl = parameters['impl']
        return detect_ray_trafo(n_voxels, impl, device, geometry)
    else:
        raise NotImplementedError(f'The pseudo inverse is not implemented for {name}. Currently, only ["backprojection"] is supported')
    
def load_pseudo_inverse_as_module(
        name:str, 
        parameters:Dict, 
        geometry:Geometry,
        device,
        ):
    pseudo_inverse = load_pseudo_inverse(
        name, parameters, geometry, device
    )
    return OperatorModule(pseudo_inverse)