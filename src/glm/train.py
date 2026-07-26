from pathlib import Path

import yaml
import torch
from torch_geometric.data import Batch
from dvclive import Live
from statistics import mean

from glm.utils import plot_image_live
from glm.dataset import parse_dataloader
from glm.models.utils import (get_angles_list_from_downsampling, load_model, load_graph, load_geometry, load_pseudo_inverse_as_module, PSNR, set_data_shape)

def training_loop():

    # We load the different parameters
    parameters = yaml.safe_load(open("params.yaml"))
    data_parameters = parameters['data']
    train_parameters = parameters['train_parameters']
    pretrain_parameters = parameters['pretrain_parameters']
    # What are the training hyperparameters
    hyperparameters = train_parameters['hyperparameters']

    torch.manual_seed(parameters['seed'])

    # Instanciate the device object
    device = torch.device(f'cuda:0')
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
        num_workers=hyperparameters['num_workers']
    )
    validation_dataloader = parse_dataloader(
        dataset_path = data_parameters['processed_path'],
        mode = 'validation',
        data_tuples=[
            ('preprocessed_sinogram', 'mode2'),
            ('preprocessed_reconstruction', 'mode2')
            ],
        batch_size=hyperparameters['batch_size'],
        num_workers=hyperparameters['num_workers']
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

    # The dataloaders use drop_last=True, so batch_size is constant across
    # every iteration below: the batched graph can be built once and reused.
    graphs = None
    if graph is not None:
        graphs = Batch.from_data_list(
            [graph for _ in range(hyperparameters['batch_size'])]
            ).to(device)

    live = Live(save_dvc_exp=True, dir="dvclive/training")


    print(f'Running experiments on device {device}')
    live.log_params(parameters)

    for epoch in range(epochs):
        
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

            current_psnr = psnr(infered_image, target_reconstruction)
            live.log_metric("PSNR", current_psnr.item())
            live.log_metric("MSE loss", loss.item())
            live.next_step()
        
            if index %50==0:
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
            live.log_metric("Validation PSNR", mean(validation))
            
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


    live.log_artifact(
        str(model_save_path), 
        type="model", 
        name="end_to_end_model",
        desc="Learned reconstruction model with sinogram and image modalities processing",
        labels=["sinogram", "image", "end-to-end", "learned reconstruction"],
        meta=parameters
        )
    torch.save({
        'sinogram_model':sinogram_model.state_dict(),
        'image_model':image_model.state_dict()
        }, model_save_path)

if __name__ == '__main__':
    training_loop()