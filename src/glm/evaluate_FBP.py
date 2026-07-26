from pathlib import Path
import os 
import sys
import signal
import torch.multiprocessing as mp

import yaml
import torch
import torch.distributed as dist
from torch_geometric.data import Batch
from ignite.metrics import PSNR, SSIM
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events

import pandas as pd 

from dvclive.live import Live
from torchvision.utils import save_image, make_grid

from glm.utils import plot_image_live
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
    # 1) The pseudo inverse
    active_pseudo_inverse = train_parameters['active_pseudo_inverse']
    pseudo_inverse_parameters = train_parameters['pseudo_inverse'][active_pseudo_inverse]
    pseudo_inverse : torch.nn.Module = load_pseudo_inverse_as_module(
        active_pseudo_inverse, pseudo_inverse_parameters, geometry, device
    )

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
        input_sinogram = batch['preprocessed_sinogram_mode2'].float().to(device)
        target_reconstruction = batch['preprocessed_reconstruction_mode2'].float().to(device)

        input_sinogram = input_sinogram[:,:,::downsampling, :]

        infered_image = pseudo_inverse(input_sinogram)
        
        infered_image = normalise(infered_image)
        target_reconstruction = normalise(target_reconstruction)
        
        return infered_image, target_reconstruction
          
    live = Live(save_dvc_exp=True, dir="dvclive/evaluate_FBP")
    
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