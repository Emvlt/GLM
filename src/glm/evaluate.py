from pathlib import Path
import os 
import sys
import signal
import torch.multiprocessing as mp

import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Batch
from ignite.metrics import PSNR, SSIM
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events

import pandas as pd 

from dvclive import Live
from torchvision.utils import save_image, make_grid

from glm.utils import plot_image_live #, setup_distributed, cleanup_distributed, signal_handler
from glm.dataset import parse_dataloader
from glm.models.utils import (get_angles_list_from_downsampling, load_model, load_graph, load_geometry, load_pseudo_inverse_as_module, set_data_shape)

def normalise(x:torch.Tensor) -> torch.Tensor:
    return (x-x.min()) / (x.max()-x.min())

def evaluate_loop():
    # We load the different parameters
    parameters = yaml.safe_load(open("params.yaml"))
    data_parameters = parameters['data']
    train_parameters = parameters['train_parameters']
    pretrain_parameters = parameters['pretrain_parameters']
    evaluate_parameters = parameters['evaluate_parameters']

    # Set the seed for reproducibility
    torch.manual_seed(parameters['seed'])

    # Instanciate the device object
    device = torch.device(f'cuda:0')

    print('Setting up evaluation')
    print(f'\t device: {device}')

    # We load the geometry object
    downsampling = evaluate_parameters['downsampling']
    angles_indices = get_angles_list_from_downsampling(downsampling)
    n_measurements = 3600 if angles_indices is None else len(angles_indices)
    geometry = load_geometry(angles_indices)

    # Now the models
    # 1) The sinogram model
    active_sinogram_model = pretrain_parameters['active_model']
    model_parameters = pretrain_parameters['models'][active_sinogram_model]
    sinogram_model : torch.nn.Module = load_model(
        active_sinogram_model, model_parameters
        )
    sinogram_model.load_state_dict(
        torch.load('src/glm/saved_models/model.pt', map_location=device)['sinogram_model'])
    sinogram_model = sinogram_model.to(device)

    # 2) The pseudo inverse
    active_pseudo_inverse = train_parameters['active_pseudo_inverse']
    pseudo_inverse_parameters = train_parameters['pseudo_inverse'][active_pseudo_inverse]
    pseudo_inverse : torch.nn.Module = load_pseudo_inverse_as_module(
        active_pseudo_inverse, pseudo_inverse_parameters, geometry, device
    )

    # 3) The image model
    active_image_model = train_parameters['active_image_model']
    model_parameters = train_parameters['image_models'][active_image_model]
    image_model : torch.nn.Module = load_model(
        active_image_model, model_parameters
        )
    image_model.load_state_dict(
        torch.load('src/glm/saved_models/model.pt', map_location=device)['image_model'])
    image_model = image_model.to(device)

    # And the graph
    graph = load_graph(active_sinogram_model, geometry)
    if graph is not None:
        print(graph)

    # Dataset
    test_dataloader = parse_dataloader(
        dataset_path = data_parameters['processed_path'],
        mode = 'testing',
        data_tuples=[
            ('preprocessed_sinogram', 'mode2'),
            ('preprocessed_reconstruction', 'mode2')
            ],
        batch_size=evaluate_parameters['batch_size'],
        num_workers=evaluate_parameters['num_workers']
    )

    def eval_step(engine, batch):
        batch_size = batch['preprocessed_sinogram_mode2'].size(0)

        input_sinogram = batch['preprocessed_sinogram_mode2'].float().to(device)
        target_reconstruction = batch['preprocessed_reconstruction_mode2'].float().to(device)

        input_sinogram  = set_data_shape(
            model = sinogram_model,
            batch_size=batch_size,
            angles_indices = angles_indices,
            n_measurements = n_measurements,
            tensor = input_sinogram, 
            target='NN')

        if graph is None:
            infered_sinogram = sinogram_model(input_sinogram)
        else:
            graphs = Batch.from_data_list([graph for sample_index in range(batch_size)] ).to(device)
            infered_sinogram = sinogram_model(input_sinogram, graphs.edge_index, graphs.edge_weight)

        infered_sinogram = set_data_shape(
            model = sinogram_model,
            batch_size = batch_size,
            angles_indices = angles_indices,
            n_measurements = n_measurements,
            tensor = infered_sinogram, 
            target = 'tomo')
        
        infered_reconstruction = pseudo_inverse(infered_sinogram)
        
        infered_image = image_model(infered_reconstruction)
        
        infered_image = normalise(infered_image)
        target_reconstruction = normalise(target_reconstruction)
        
        return infered_image, target_reconstruction
     
    live = Live(save_dvc_exp=True, dir="dvclive/evaluate") 
    
    sinogram_model.eval()
    image_model.eval()
    
    evaluator = Engine(eval_step)
    ssim = SSIM(data_range=1.0) 
    psnr = PSNR(data_range=1.0)
    ssim.attach(evaluator, 'ssim')
    psnr.attach(evaluator, 'psnr')
    
    pbar = ProgressBar()
    pbar.attach(evaluator)
    
    # @evaluator.on(Events.COMPLETED)
    # def save_batch_images(engine):
        
    #     y_pred, y_true = engine.state.output
        
    #     batch_size = y_pred.size(0)
        
    #     combined = torch.cat((y_pred, y_true), dim=0)
        
    #     grid = make_grid(combined, nrow=batch_size, padding=2, normalize=False)
    
    #     # 4. Save to disk using the current iteration number
    #     save_image(grid, f"src/glm/test_images_{downsampling}.png")
    
    # Run the Engine
    state = evaluator.run(test_dataloader)

    # Get the Result
    test_psnr = state.metrics['psnr']
    test_ssim = state.metrics['ssim']
    print(f"Computed SSIM: {test_ssim:.4f}")
    print(f"Computed PSNR: {test_psnr:.4f}")
    
    live.log_metric("Test SSIM", test_ssim)
    live.log_metric("Test PSNR", test_psnr)
    live.log_param('Downsampling', downsampling)
        

if __name__ == '__main__':

    evaluate_loop()