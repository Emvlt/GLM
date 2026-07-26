import argparse

import torch
from ignite.metrics import PSNR, SSIM
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine

from dvclive import Live

from glm.config import load_params, init_run, COMBINED_MODEL_PATH, log_ignite_metrics
from glm.dataset import parse_dataloader
from glm.models.utils import (
    load_model, load_graph, build_geometry, build_batched_graph,
    build_pseudo_inverse_from_params, forward_sinogram_model,
    )

def normalise(x:torch.Tensor) -> torch.Tensor:
    return (x-x.min()) / (x.max()-x.min())

def evaluate_loop(downsampling: int | None = None):
    # We load the different parameters
    parameters = load_params()
    data_parameters = parameters['data']
    train_parameters = parameters['train_parameters']
    pretrain_parameters = parameters['pretrain_parameters']
    evaluate_parameters = parameters['evaluate_parameters']

    device = init_run(parameters['seed'])

    print('Setting up evaluation')
    print(f'\t device: {device}')

    # We load the geometry object
    if downsampling is None:
        downsampling = evaluate_parameters['downsampling']
    geo = build_geometry(downsampling)
    angles_indices, n_measurements, geometry = geo.angles_indices, geo.n_measurements, geo.geometry

    # Now the models
    # 1) The sinogram model
    active_sinogram_model = pretrain_parameters['active_model']
    model_parameters = pretrain_parameters['models'][active_sinogram_model]
    sinogram_model : torch.nn.Module = load_model(
        active_sinogram_model, model_parameters
        )
    sinogram_model.load_state_dict(
        torch.load(COMBINED_MODEL_PATH, map_location=device)['sinogram_model'])
    sinogram_model = sinogram_model.to(device)

    # 2) The pseudo inverse
    pseudo_inverse : torch.nn.Module = build_pseudo_inverse_from_params(
        train_parameters, geometry, device
    )

    # 3) The image model
    active_image_model = train_parameters['active_image_model']
    model_parameters = train_parameters['image_models'][active_image_model]
    image_model : torch.nn.Module = load_model(
        active_image_model, model_parameters
        )
    image_model.load_state_dict(
        torch.load(COMBINED_MODEL_PATH, map_location=device)['image_model'])
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

    # The dataloader uses drop_last=True, so batch_size is constant across
    # every batch below: the batched graph can be built once and reused.
    graphs = build_batched_graph(graph, evaluate_parameters['batch_size'], device)

    def eval_step(engine, batch):
        batch_size = batch['preprocessed_sinogram_mode2'].size(0)

        input_sinogram = batch['preprocessed_sinogram_mode2'].float().to(device)
        target_reconstruction = batch['preprocessed_reconstruction_mode2'].float().to(device)

        _, infered_sinogram = forward_sinogram_model(
            sinogram_model, input_sinogram,
            batch_size = batch_size,
            angles_indices = angles_indices,
            n_measurements = n_measurements,
            graph = graph,
            graphs = graphs,
            reshape_to_tomo = True,
            )

        infered_reconstruction = pseudo_inverse(infered_sinogram)

        infered_image = image_model(infered_reconstruction)

        infered_image = normalise(infered_image)
        target_reconstruction = normalise(target_reconstruction)

        return infered_image, target_reconstruction

    live = Live(save_dvc_exp=True, dir=f"dvclive/evaluate/{downsampling}")

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

    log_ignite_metrics(evaluator, test_dataloader, live, downsampling)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--downsampling', type=int, default=None,
        help='overrides evaluate_parameters.downsampling from params.yaml',
        )
    args = parser.parse_args()

    evaluate_loop(args.downsampling)
