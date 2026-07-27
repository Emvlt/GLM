from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple, Any

import torch
from torch_geometric.nn import Sequential
from torch_geometric.data import Batch

from odl.contrib.torch.operator import OperatorModule
from odl.applications.tomo.geometry import Geometry
from odl.applications.tomo.analytic import fbp_op
from odl.contrib.graphs.graph_interface import create_graph_from_geometry
from odl.contrib.datasets.ct.detect import detect_geometry, detect_ray_trafo

from .cnn import ImageCNN, CNN_Module
from .gnn import GLM_Module

TOTAL_PROJECTION_ANGLES = 3600
N_PIXELS = 956

class PSNR:
    def __init__(self, reduction = 'mean'):
        self.loss = torch.nn.MSELoss(reduction=reduction)

    def __call__(self, inferred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 20 * torch.log10(target.max() - target.min()) - 10 * torch.log10(
            self.loss(inferred, target)
        )

DEFAULT_N_MODULES = 3

def _module_channel_sizes(n_channels: int, n_modules: int) -> List[Tuple[int, int]]:
    """Returns the (in_channels, out_channels) pairs for a stack of n_modules layers
    going from 1 channel, through n_modules - 1 hidden layers of n_channels, back to 1 channel.
    """
    if n_modules < 1:
        raise ValueError(f'n_modules must be >= 1, got {n_modules}')
    sizes = [1] + [n_channels] * (n_modules - 1) + [1]
    return list(zip(sizes[:-1], sizes[1:]))

def _build_sinogram_cnn(model_parameters: Dict) -> torch.nn.Module:
    n_channels = model_parameters['n_channels']
    n_modules = int(model_parameters.get('n_modules', DEFAULT_N_MODULES))

    layers = []
    for in_channels, out_channels in _module_channel_sizes(n_channels, n_modules):
        layers.append(CNN_Module(in_channels, out_channels, model_parameters))
        layers.append(torch.nn.LeakyReLU(negative_slope=0.1, inplace=True))
    return torch.nn.Sequential(*layers)

def _build_image_cnn(model_parameters: Dict) -> torch.nn.Module:
    return ImageCNN(model_parameters)

def _build_glm(model_parameters: Dict) -> torch.nn.Module:
    n_channels = model_parameters['n_channels']
    n_modules = int(model_parameters.get('n_modules', DEFAULT_N_MODULES))

    layers = []
    for in_channels, out_channels in _module_channel_sizes(n_channels, n_modules):
        layers.append((
            GLM_Module(in_channels, out_channels, model_parameters),
            'x, edge_index, edge_weight -> x'
            ))
        layers.append(torch.nn.LeakyReLU(negative_slope=0.1, inplace=True))
    return Sequential('x, edge_index, edge_weight', layers)

MODEL_REGISTRY: Dict[str, Callable[[Dict], torch.nn.Module]] = {
    'sinogram_CNN': _build_sinogram_cnn,
    'image_CNN': _build_image_cnn,
    'GLM': _build_glm,
}

def load_model(model_name:str, model_parameters:Dict):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'The model {model_name} does not have an implementation.')
    return MODEL_REGISTRY[model_name](model_parameters)

def load_geometry(angles_indices:List[int]=None):
    return detect_geometry(n_voxels = 1024, angles_indices=angles_indices)

def load_graph(
        model_name,
        geometry,
        connectivity: Optional[int] = None,
        kernel_type: Optional[str] = None,
        sigma: Optional[float] = None,
        distance_mode: Optional[str] = None,
        ):
    if model_name == 'GLM':
        edges_kwargs = {} if connectivity is None else {'connectivity': connectivity}
        weights_kwargs = {}
        if kernel_type is not None:
            weights_kwargs['kernel_type'] = kernel_type
        if sigma is not None:
            weights_kwargs['sigma'] = sigma
        if distance_mode is not None:
            weights_kwargs['distance_mode'] = distance_mode
        return create_graph_from_geometry(
            geometry = geometry,
            scheme   = 'GLM',
            backend  = 'torch_geometric',
            edges_kwargs = edges_kwargs,
            weights_kwargs = weights_kwargs,
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
        n_pixels : int = N_PIXELS,
        ):
    if isinstance(model, torch.nn.Sequential) or isinstance(model, Sequential):
        return model[0].set_data_shape(
            batch_size, angles_indices, n_measurements, n_pixels, tensor, target
            )
    else:
        return model.set_data_shape(
            batch_size, angles_indices, n_measurements, n_pixels, tensor, target
            )

def get_angles_list_from_downsampling(downsampling : int = 1):
    assert isinstance(downsampling, int)
    if downsampling==1:
        return None
    return [i for i in range(0, TOTAL_PROJECTION_ANGLES, downsampling)]

@dataclass
class GeometryConfig:
    angles_indices: Optional[List[int]]
    n_measurements: int
    geometry: Geometry

def build_geometry(downsampling: int) -> GeometryConfig:
    angles_indices = get_angles_list_from_downsampling(downsampling)
    n_measurements = TOTAL_PROJECTION_ANGLES if angles_indices is None else len(angles_indices)
    geometry = load_geometry(angles_indices)
    return GeometryConfig(angles_indices, n_measurements, geometry)

def build_batched_graph(graph, batch_size: int, device) -> Optional[Batch]:
    if graph is None:
        return None
    return Batch.from_data_list([graph for _ in range(batch_size)]).to(device)

def forward_sinogram_model(
        model : torch.nn.Sequential | Sequential,
        input_sinogram : torch.Tensor,
        *,
        batch_size : int,
        angles_indices : None | List,
        n_measurements : int,
        graph,
        graphs,
        reshape_to_tomo : bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    input_sinogram = set_data_shape(
        model = model,
        batch_size = batch_size,
        angles_indices = angles_indices,
        n_measurements = n_measurements,
        tensor = input_sinogram,
        target = 'NN')

    if graph is None:
        infered_sinogram = model(input_sinogram)
    else:
        infered_sinogram = model(input_sinogram, graphs.edge_index, graphs.edge_weight)

    if reshape_to_tomo:
        infered_sinogram = set_data_shape(
            model = model,
            batch_size = batch_size,
            angles_indices = angles_indices,
            n_measurements = n_measurements,
            tensor = infered_sinogram,
            target = 'tomo')

    return input_sinogram, infered_sinogram

def _build_backprojection(parameters: Dict, geometry: Geometry, device) -> Any:
    n_voxels = parameters['n_voxels']
    impl = parameters['impl']
    return detect_ray_trafo(n_voxels, impl, device, geometry).adjoint

def _build_filtered_backprojection(parameters: Dict, geometry: Geometry, device) -> Any:
    n_voxels = parameters['n_voxels']
    impl = parameters['impl']
    ray_trafo = detect_ray_trafo(n_voxels, impl, device, geometry)
    filter_type = parameters['filter_type'] if 'filter_type' in parameters else 'Ram-Lak'
    frequency_scaling = parameters['frequency_scaling'] if 'frequency_scaling' in parameters else 1.0
    padding = parameters['padding'] if 'padding' in parameters else True
    return fbp_op(ray_trafo, padding, filter_type, frequency_scaling)

PSEUDO_INVERSE_REGISTRY: Dict[str, Callable[[Dict, Geometry, Any], Any]] = {
    'backprojection': _build_backprojection,
    'filtered_backprojection': _build_filtered_backprojection,
}

def load_pseudo_inverse(
        name:str,
        parameters:Dict,
        geometry:Geometry,
        device,
        ):
    if name not in PSEUDO_INVERSE_REGISTRY:
        raise NotImplementedError(f'The pseudo inverse is not implemented for {name}. Currently, only ["backprojection", "filtered_backprojection"] is supported')
    return PSEUDO_INVERSE_REGISTRY[name](parameters, geometry, device)

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

def build_pseudo_inverse_from_params(train_parameters: Dict, geometry: Geometry, device) -> torch.nn.Module:
    active_pseudo_inverse = train_parameters['active_pseudo_inverse']
    pseudo_inverse_parameters = train_parameters['pseudo_inverse'][active_pseudo_inverse]
    return load_pseudo_inverse_as_module(
        active_pseudo_inverse, pseudo_inverse_parameters, geometry, device
    )
