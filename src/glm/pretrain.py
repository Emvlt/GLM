from pathlib import Path
import argparse

import json
import yaml
import torch
from torch_geometric.data import Batch
from dvclive import Live
from statistics import mean

from glm.utils import plot_image_live
from glm.dataset import parse_dataloader
from glm.models.utils import (get_angles_list_from_downsampling, load_model, load_graph, load_geometry, PSNR, set_data_shape)

def pretraining_loop():
    # We load the different parameters
    data_parameters = yaml.safe_load(open("params.yaml"))['data']
    parameters = yaml.safe_load(open("params.yaml"))['pretrain_parameters']
    # What are the training hyperparameters
    hyperparameters = parameters['hyperparameters']
    # Instanciate the device object
    device = torch.device(hyperparameters['device'])
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
    # And the graph
    graph = load_graph(active_model, geometry)
    if graph is not None:
        graph = graph.to(device)
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
        num_workers=hyperparameters['num_workers']
    )
    validation_dataloader = parse_dataloader(
        dataset_path = data_parameters['processed_path'],
        mode = 'validation',
        data_tuples=[('preprocessed_sinogram', 'mode2')],
        batch_size=hyperparameters['batch_size'],
        num_workers=hyperparameters['num_workers']
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

    with Live(save_dvc_exp=True, dir="dvclive") as live:

        live.log_params(hyperparameters)

        for epoch in range(epochs):
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
                    graphs = Batch.from_data_list([graph for sample_index in range(batch_size)] )
                    infered_sinogram = model(input_sinogram, graphs.edge_index, graphs.edge_weight)
                
                loss = loss_function(infered_sinogram, input_sinogram)
                loss.backward()
                optimiser.step()

                current_psnr = psnr(infered_sinogram, input_sinogram)
                live.log_metric(
                    "pretraining/training/PSNR", current_psnr.item())
                live.log_metric("pretraining/training/loss", loss.item())
                live.next_step()
            
                if index %50==0:
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
        

        live.log_metric("pretraining/validation/PSNR_loss", mean(validation))
        live.log_artifact(str(model_save_path), type="model", name="pretrained_sinogram_model")
        torch.save(model, model_save_path)
        

if __name__ == '__main__':
    pretraining_loop()