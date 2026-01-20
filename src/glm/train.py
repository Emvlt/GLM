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
import torch_geometric
from dvclive import Live
from statistics import mean

from glm.utils import plot_image_live, setup_distributed, cleanup_distributed, signal_handler
from glm.dataset import parse_dataloader
from glm.models.utils import (get_angles_list_from_downsampling, load_model, load_graph, load_geometry, load_pseudo_inverse_as_module, PSNR, set_data_shape)

def training_loop():

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    # We load the different parameters
    parameters = yaml.safe_load(open("params.yaml"))
    data_parameters = parameters['data']
    train_parameters = parameters['train_parameters']
    pretrain_parameters = parameters['pretrain_parameters']
    # What are the training hyperparameters
    hyperparameters = train_parameters['hyperparameters']

    # Instanciate the device object
    local_rank = int(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    print('Seting up distributed learning:')
    print(f'\t rank: {rank}')
    print(f'\t local rank: {local_rank}')
    print(f'\t world size: {world_size}')
    print(f'\t device: {device}')

    # We load the geometry object
    downsampling = hyperparameters['downsampling']
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
        torch.load('src/glm/saved_models/pretrained_sinogram_model.pt', map_location=device, weights_only=True))
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
    image_model = image_model.to(device)

    # Wrap model in DDP if using multiple GPUs
    if world_size > 1:
        sinogram_model = DDP(sinogram_model, device_ids=[local_rank], output_device=local_rank)
        image_model = DDP(image_model, device_ids=[local_rank], output_device=local_rank)

    # And the graph
    graph = load_graph(active_sinogram_model, geometry)
    if graph is not None:
        print(graph)

    # Loss functions
    psnr = PSNR()
    loss_function = torch.nn.MSELoss()

    # Datasets
    train_dataloader = parse_dataloader(
        dataset_path = data_parameters['processed_path'],
        mode = 'training',
        data_tuples=[
            ('preprocessed_sinogram', 'mode2'),
            ('preprocessed_reconstruction', 'mode2')
            ],
        batch_size=hyperparameters['batch_size'],
        num_workers=0,
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size
    )
    validation_dataloader = parse_dataloader(
        dataset_path = data_parameters['processed_path'],
        mode = 'validation',
        data_tuples=[
            ('preprocessed_sinogram', 'mode2'),
            ('preprocessed_reconstruction', 'mode2')
            ],
        batch_size=hyperparameters['batch_size'],
        num_workers=hyperparameters['num_workers'],
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size
    )

    ### Unpacking string hyperparameters
    learning_rate = float(hyperparameters['learning_rate'])
    epochs = int(hyperparameters['epochs'])
    optimiser = torch.optim.Adam(
        list(sinogram_model.parameters()) + list(image_model.parameters()),
        lr=learning_rate
        )
    
    model_save_path = Path('src/glm/saved_models/model.pt')
    model_save_path.parent.mkdir(exist_ok=True)

    live = Live(save_dvc_exp=True, dir="dvclive") if is_main_process else None

    try:
        print(f'Running experiments on device {device}')
        if is_main_process:
            live.log_params(parameters)

        for epoch in range(epochs):

            if world_size > 1:
                train_dataloader.sampler.set_epoch(epoch)
            
            sinogram_model.train()
            image_model.train()
            for index, tensor_dict in enumerate(train_dataloader):
                batch_size = tensor_dict['preprocessed_sinogram_mode2'].size(0)

                input_sinogram = tensor_dict['preprocessed_sinogram_mode2'].float().to(device)
                target_reconstruction = tensor_dict['preprocessed_reconstruction_mode2'].float().to(device)

                input_sinogram  = set_data_shape(
                    model = sinogram_model,
                    batch_size=batch_size,
                    angles_indices = angles_indices,
                    n_measurements = n_measurements,
                    tensor = input_sinogram, 
                    target='NN')

                optimiser.zero_grad()

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
                
                loss = loss_function(infered_image, target_reconstruction)
                loss.backward()
                optimiser.step()

                if is_main_process:
                    current_psnr = psnr(infered_image, target_reconstruction)
                    live.log_metric(
                        "training/PSNR", current_psnr.item())
                    live.log_metric("training/loss", loss.item())
                    live.next_step()
            
                if is_main_process and index %50==0:
                    plot_image_live(
                    data = infered_image, 
                    name = 'infered_image',
                    title='Infered Image',
                    extension='jpg',
                    live_session = live
                    )

                    plot_image_live(
                    data = target_reconstruction, 
                    name = 'target_reconstruction',
                    title='Target Image',
                    extension='jpg',
                    live_session = live
                    )

        validation = []
        sinogram_model.eval()
        image_model.eval()
        with torch.no_grad():
            for index, tensor_dict in enumerate(validation_dataloader):
                batch_size = tensor_dict['preprocessed_sinogram_mode2'].size(0)

                input_sinogram = tensor_dict['preprocessed_sinogram_mode2'].float().to(device)
                target_reconstruction = tensor_dict['preprocessed_reconstruction_mode2'].float().to(device)

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
                
                validation.append(
                    psnr(infered_image, target_reconstruction).item()
                    )
                
        if world_size > 1:
            validation_tensor = torch.tensor(validation, device=device)
            gathered = [torch.zeros_like(validation_tensor) for _ in range(world_size)]
            dist.all_gather(gathered, validation_tensor)
            if is_main_process:
                all_validation = torch.cat(gathered).cpu().tolist()
                validation = all_validation
            plot_image_live(
                    data = infered_image, 
                    name = 'infered_image_validation',
                    title='Infered Image',
                    extension='jpg',
                    live_session = live
                    )

            plot_image_live(
                    data = target_reconstruction, 
                    name = 'target_reconstruction_validation',
                    title='Target Image',
                    extension='jpg',
                    live_session = live
                    )

        if is_main_process:
            live.log_metric("pretraining/validation/PSNR_loss", mean(validation))
            live.log_artifact(
                str(model_save_path), 
                type="model", 
                name="end_to_end_model",
                desc="Learned reconstruction model with sinogram and image modalities processing",
                labels=["sinogram", "image", "end-to-end", "learned reconstruction"],
                meta=parameters
                )
            sin_model_to_save = sinogram_model.module if world_size > 1 else sinogram_model
            img_model_to_save = image_model.module if world_size > 1 else  image_model
            torch.save({
                'sinogram_model':sin_model_to_save.state_dict(),
                'image_model':img_model_to_save.state_dict()
                }, model_save_path)

    finally:
        if is_main_process and live is not None:
            live.end()
        cleanup_distributed()   

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    training_loop()