import argparse

import torch
from ignite.metrics import PSNR, SSIM
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine

from dvclive import Live

from glm.config import load_params, init_run, SINOGRAM_DENOISING_MODEL_PATH, SINOGRAM_DENOISING_PARAMS_PATH, log_ignite_metrics
from glm.dataset import parse_dataloader
from glm.models.utils import (
    load_model, load_graph, build_geometry, build_batched_graph,
    forward_sinogram_model,
    )

def normalise(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return (x - reference.min()) / (reference.max() - reference.min())

def evaluate_loop(params_path: str = SINOGRAM_DENOISING_PARAMS_PATH, downsampling: int | None = None):
    # We load the different parameters
    parameters = load_params(params_path)
    data_parameters = parameters['data']
    hyperparameters = parameters['hyperparameters']
    model_parameters = parameters['model_parameters']

    device = init_run(parameters['seed'])

    print('Setting up sinogram denoising evaluation')
    print(f'\t device: {device}')

    # We load the geometry object
    if downsampling is None:
        downsampling = hyperparameters['downsampling']
    geo = build_geometry(downsampling)
    angles_indices, n_measurements, geometry = geo.angles_indices, geo.n_measurements, geo.geometry

    # Now the model
    active_model = model_parameters['active_model']
    active_model_parameters = model_parameters['models'][active_model]
    model : torch.nn.Module = load_model(active_model, active_model_parameters)
    model.load_state_dict(
        torch.load(SINOGRAM_DENOISING_MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)

    # And the graph
    graph_kwargs = active_model_parameters.get('graph_kwargs')
    graph = load_graph(active_model, geometry, **(graph_kwargs or {}))
    if graph is not None:
        print(graph)

    # Dataset
    input_mode  = hyperparameters['input_mode']
    target_mode = hyperparameters['target_mode']
    batch_size  = hyperparameters['batch_size']
    test_dataloader = parse_dataloader(
        dataset_path = data_parameters['processed_path'],
        mode = 'testing',
        data_tuples=[
            ('preprocessed_sinogram', input_mode),
            ('preprocessed_sinogram', target_mode)
            ],
        batch_size=batch_size,
        num_workers=hyperparameters['num_workers']
    )

    # The dataloader uses drop_last=True, so batch_size is constant across
    # every batch below: the batched graph can be built once and reused.
    graphs = build_batched_graph(graph, batch_size, device)

    def eval_step(engine, batch):
        batch_size = batch[f'preprocessed_sinogram_{input_mode}'].size(0)

        input_sinogram = batch[f'preprocessed_sinogram_{input_mode}'].float().to(device)
        target_sinogram = batch[f'preprocessed_sinogram_{target_mode}'].float().to(device)
        if angles_indices is not None:
            target_sinogram = target_sinogram[:, :, angles_indices, :]

        _, infered_sinogram = forward_sinogram_model(
            model, input_sinogram,
            batch_size = batch_size,
            angles_indices = angles_indices,
            n_measurements = n_measurements,
            graph = graph,
            graphs = graphs,
            reshape_to_tomo = True,
            )

        reference = target_sinogram
        infered_sinogram = normalise(infered_sinogram, reference)
        target_sinogram = normalise(target_sinogram, reference)

        return infered_sinogram, target_sinogram

    live = Live(save_dvc_exp=True, dir=f"dvclive/evaluate_sinogram_denoising/{downsampling}")

    model.eval()

    evaluator = Engine(eval_step)
    ssim = SSIM(data_range=1.0)
    psnr = PSNR(data_range=1.0)
    ssim.attach(evaluator, 'ssim')
    psnr.attach(evaluator, 'psnr')

    pbar = ProgressBar()
    pbar.attach(evaluator)

    log_ignite_metrics(evaluator, test_dataloader, live, downsampling)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--params', type=str, default=SINOGRAM_DENOISING_PARAMS_PATH,
        help='Path to the yaml file holding the sinogram denoising experiment parameters.',
        )
    parser.add_argument(
        '--downsampling', type=int, default=None,
        help='Overrides hyperparameters.downsampling from the params yaml.',
        )
    args = parser.parse_args()

    evaluate_loop(args.params, args.downsampling)
