import argparse
import json

import torch
from dvclive import Live
from statistics import mean

from glm.config import load_params, init_run, SINOGRAM_DENOISING_MODEL_PATH, SINOGRAM_DENOISING_PARAMS_PATH
from glm.utils import plot_image_live, plot_image
from glm.dataset import parse_dataloader
from glm.models.utils import (
    load_model, load_graph, build_geometry, build_batched_graph,
    forward_sinogram_model, PSNR, N_PIXELS
    )


def training_loop(
        params_path: str = SINOGRAM_DENOISING_PARAMS_PATH,
        ):

    # We load the different parameters
    parameters = load_params(params_path)

    print(json.dumps(parameters, indent=4))

    data_parameters = parameters['data']
    # What are the training hyperparameters
    hyperparameters  = parameters['hyperparameters']
    model_parameters = parameters['model_parameters']

    device = init_run(parameters['seed'])

    # We load the geometry object
    downsampling = hyperparameters['downsampling']
    geo = build_geometry(downsampling)
    angles_indices, n_measurements, geometry = geo.angles_indices, geo.n_measurements, geo.geometry

    # Now the model
    active_model = model_parameters['active_model']
    model_parameters = model_parameters['models'][active_model]
    model : torch.nn.Module = load_model(active_model, model_parameters)
    model = model.to(device)

    # And the graph
    graph_kwargs = model_parameters.get('graph_kwargs')
    graph = load_graph(active_model, geometry, **(graph_kwargs or {}))
    if graph is not None:
        print(graph)

    # Loss functions
    psnr = PSNR()
    loss_function = torch.nn.MSELoss()

    # Datasets
    input_mode  = hyperparameters['input_mode']
    target_mode = hyperparameters['target_mode']
    batch_size  = hyperparameters['batch_size']
    train_dataloader = parse_dataloader(
        dataset_path = data_parameters['processed_path'],
        mode = 'training',
        data_tuples=[
            ('preprocessed_sinogram', input_mode),
            ('preprocessed_sinogram', target_mode)
            ],
        batch_size=batch_size,
        num_workers=hyperparameters['num_workers']
    )
    validation_dataloader = parse_dataloader(
        dataset_path = data_parameters['processed_path'],
        mode = 'validation',
        data_tuples=[
            ('preprocessed_sinogram', input_mode),
            ('preprocessed_sinogram', target_mode)
        ],
        batch_size=batch_size,
        num_workers=hyperparameters['num_workers']
    )

    ### Unpacking string hyperparameters
    learning_rate = float(hyperparameters['learning_rate'])
    epochs = int(hyperparameters['epochs'])
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
        )

    # Early stopping / checkpointing hyperparameters
    early_stopping_patience = int(hyperparameters.get('early_stopping_patience', 5))
    early_stopping_min_delta = float(hyperparameters.get('early_stopping_min_delta', 0.0))

    model_save_path = SINOGRAM_DENOISING_MODEL_PATH
    live_dir = "dvclive/sinogram_denoising"
    artifact_name = "sinogram_denoising_model"
    
    model_save_path.parent.mkdir(exist_ok=True)

    # The dataloaders use drop_last=True, so batch_size is constant across
    # every iteration below: the batched graph can be built once and reused.
    graphs = build_batched_graph(graph, hyperparameters['batch_size'], device)

    live = Live(save_dvc_exp=True, dir=live_dir)

    print(f'Running experiments on device {device}')
    live.log_params(hyperparameters)
    if graph_kwargs is not None:
        live.log_params({'graph': graph_kwargs})

    best_validation_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):

        model.train()
        for index, tensor_dict in enumerate(train_dataloader):

            input_sinogram = tensor_dict[f'preprocessed_sinogram_{input_mode}'].float().to(device)
            target_sinogram = tensor_dict[f'preprocessed_sinogram_{target_mode}'].float().to(device)
            optimiser.zero_grad()

            input_sinogram, infered_sinogram = forward_sinogram_model(
                model, input_sinogram,
                batch_size = batch_size,
                angles_indices = angles_indices,
                n_measurements = n_measurements,
                graph = graph,
                graphs = graphs,
                reshape_to_tomo = True,
                )

            loss = loss_function(infered_sinogram, target_sinogram)
            loss.backward()
            optimiser.step()

            if index %50==0:
                current_psnr = psnr(infered_sinogram, target_sinogram)
                live.log_metric("PSNR", current_psnr.item())
                live.log_metric("MSE loss", loss.item())
                live.next_step()

                print(f'Current PSNR: {current_psnr.item():.6f}')
                plot_image(infered_sinogram, 'test')
                
                plot_image_live(
                data = infered_sinogram.view(batch_size, n_measurements, N_PIXELS),
                name = 'infered_sinogram',
                title='Infered Sinogram',
                extension='jpg',
                live_session = live
                )

                plot_image_live(
                input_sinogram.view(batch_size, n_measurements, N_PIXELS),
                name = 'input_sinogram',
                title='Input Sinogram',
                extension='jpg',
                live_session = live
                )

        validation_losses = []
        validation_psnrs = []
        model.eval()
        with torch.no_grad():
            for index, tensor_dict in enumerate(validation_dataloader):

                input_sinogram = tensor_dict[f'preprocessed_sinogram_{input_mode}'].float().to(device)
                target_sinogram = tensor_dict[f'preprocessed_sinogram_{target_mode}'].float().to(device)

                input_sinogram, infered_sinogram = forward_sinogram_model(
                    model, input_sinogram,
                    batch_size = batch_size,
                    angles_indices = angles_indices,
                    n_measurements = n_measurements,
                    graph = graph,
                    graphs = graphs,
                    reshape_to_tomo = True,
                    )

                validation_losses.append(
                    loss_function(infered_sinogram, target_sinogram).item()
                    )
                validation_psnrs.append(
                    psnr(infered_sinogram, target_sinogram).item()
                    )

        validation_loss = mean(validation_losses)
        live.log_metric("Validation MSE loss", validation_loss)
        live.log_metric("Validation PSNR", mean(validation_psnrs))
        live.next_step()

        if validation_loss < best_validation_loss - early_stopping_min_delta:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            print(f'Epoch {epoch}: validation loss improved to {validation_loss:.6f}, saved weights to {model_save_path}')
        else:
            epochs_without_improvement += 1
            print(
                f'Epoch {epoch}: validation loss did not improve '
                f'({validation_loss:.6f}, best {best_validation_loss:.6f}), '
                f'{epochs_without_improvement}/{early_stopping_patience} epochs without improvement'
                )

            if epochs_without_improvement >= early_stopping_patience:
                print(f'Validation loss plateaued for {early_stopping_patience} epochs, stopping early at epoch {epoch}')
                break

    live.log_artifact(
        path=str(model_save_path),
        type="model",
        name=artifact_name,
        desc="Pretrained model for sinogram processing",
        labels=['sinogram', 'pretraining'],
        meta=parameters
        )

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--params', type=str, default=SINOGRAM_DENOISING_PARAMS_PATH,
        help='Path to the yaml file holding the sinogram denoising experiment parameters.'
        )
    args = arg_parser.parse_args()
    training_loop(args.params)
