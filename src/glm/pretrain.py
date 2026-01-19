from pathlib import Path
import os 

import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Batch
from dvclive import Live
from statistics import mean

from glm.utils import plot_image_live, setup_distributed, cleanup_distributed
from glm.dataset import parse_dataloader
from glm.models.utils import (get_angles_list_from_downsampling, load_model, load_graph, load_geometry, PSNR, set_data_shape)

def pretraining_loop():

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    # We load the different parameters
    data_parameters = yaml.safe_load(open("params.yaml"))['data']
    parameters = yaml.safe_load(open("params.yaml"))['pretrain_parameters']
    # What are the training hyperparameters
    hyperparameters = parameters['hyperparameters']

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

    # Now the model
    active_model = parameters['active_model']
    model_parameters = parameters['models'][active_model]
    model : torch.nn.Module = load_model(active_model, model_parameters)
    model = model.to(device)

    # Wrap model in DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # And the graph
    graph = load_graph(active_model, geometry)
    if graph is not None:
        print(graph)

    # Loss functions
    psnr = PSNR()
    loss_function = torch.nn.MSELoss()

    # Datasets
    train_dataloader = parse_dataloader(
        dataset_path = data_parameters['processed_path'],
        mode = 'training',
        data_tuples=[('preprocessed_sinogram', 'mode2')],
        batch_size=hyperparameters['batch_size'],
        num_workers=hyperparameters['num_workers'],
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size
    )
    validation_dataloader = parse_dataloader(
        dataset_path = data_parameters['processed_path'],
        mode = 'validation',
        data_tuples=[('preprocessed_sinogram', 'mode2')],
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
        model.parameters(),
        lr=learning_rate
        )
    
    model_save_path = Path('src/glm/saved_models/pretrained_sinogram_model.pt')
    model_save_path.parent.mkdir(exist_ok=True)

    live = Live(save_dvc_exp=True, dir="dvclive") if is_main_process else None

    try:
        print(f'Running experiments on device {device}')
        if is_main_process:
            live.log_params(hyperparameters)

        for epoch in range(epochs):

            if world_size > 1:
                train_dataloader.sampler.set_epoch(epoch)
            
            model.train()
            for index, tensor_dict in enumerate(train_dataloader):
                batch_size = tensor_dict['preprocessed_sinogram_mode2'].size(0)

                input_sinogram = tensor_dict['preprocessed_sinogram_mode2'].float().to(device)

                input_sinogram  = set_data_shape(
                    model = model,
                    batch_size=batch_size,
                    angles_indices = angles_indices,
                    n_measurements = n_measurements,
                    tensor = input_sinogram, 
                    target='NN')

                optimiser.zero_grad()

                if graph is None:
                    infered_sinogram = model(input_sinogram)
                else:
                    graphs = Batch.from_data_list([graph for sample_index in range(batch_size)] ).to(device)
                    infered_sinogram = model(input_sinogram, graphs.edge_index, graphs.edge_weight)

                
                loss = loss_function(infered_sinogram, input_sinogram)
                loss.backward()
                optimiser.step()

                if is_main_process:
                    current_psnr = psnr(infered_sinogram, input_sinogram)
                    live.log_metric(
                        "pretraining/training/PSNR", current_psnr.item())
                    live.log_metric("pretraining/training/loss", loss.item())
                    live.next_step()
            
                if is_main_process and index %50==0:
                    plot_image_live(
                    data = infered_sinogram.view(batch_size, n_measurements, 956), 
                    name = 'infered_sinogram',
                    title='Infered Sinogram',
                    extension='jpg',
                    live_session = live
                    )

                    plot_image_live(
                    input_sinogram.view(batch_size, n_measurements, 956), 
                    name = 'input_sinogram',
                    title='Input Sinogram',
                    extension='jpg',
                    live_session = live
                    )

        validation = []
        model.eval()
        with torch.no_grad():
            for index, tensor_dict in enumerate(validation_dataloader):
                batch_size = tensor_dict['preprocessed_sinogram_mode2'].size(0)

                input_sinogram = tensor_dict['preprocessed_sinogram_mode2'].float().to(device)

                input_sinogram  = set_data_shape(
                    model = model,
                    batch_size=batch_size,
                    angles_indices = angles_indices,
                    n_measurements = n_measurements,
                    tensor = input_sinogram, 
                    target='NN')

                if graph is None:
                    infered_sinogram = model(input_sinogram)
                else:
                    graphs = Batch.from_data_list([graph for sample_index in range(batch_size)] )
                    infered_sinogram = model(input_sinogram, graphs.edge_index, graphs.edge_weight)

                validation.append(
                    psnr(infered_sinogram, input_sinogram).item()
                    )
                
        if world_size > 1:
            validation_tensor = torch.tensor(validation, device=device)
            gathered = [torch.zeros_like(validation_tensor) for _ in range(world_size)]
            dist.all_gather(gathered, validation_tensor)
            if is_main_process:
                all_validation = torch.cat(gathered).cpu().tolist()
                validation = all_validation        

        if is_main_process:
            live.log_metric("pretraining/validation/PSNR_loss", mean(validation))
            live.log_artifact(
                path=str(model_save_path), 
                type="model", 
                name="pretrained_sinogram_model",
                desc="Pretrained model for sinogram processing",
                labels=['sinogram', 'pretraining'],
                meta=parameters
                )
            model_to_save = model.module if world_size > 1 else model
            torch.save(model_to_save.state_dict(), model_save_path)

    finally:
        if is_main_process and live is not None:
            live.end()
        cleanup_distributed()

        

if __name__ == '__main__':
    pretraining_loop()