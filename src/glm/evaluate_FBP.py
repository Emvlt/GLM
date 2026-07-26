import torch
from ignite.metrics import PSNR, SSIM
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine

from dvclive.live import Live

from glm.config import load_params, init_run, log_ignite_metrics
from glm.dataset import parse_dataloader
from glm.models.utils import build_geometry, build_pseudo_inverse_from_params

def normalise(x:torch.Tensor) -> torch.Tensor:
    return (x-x.min()) / (x.max()-x.min())

def evaluate_loop():
    # We load the different parameters
    parameters = load_params()
    data_parameters = parameters['data']
    train_parameters = parameters['train_parameters']
    evaluate_parameters = parameters['evaluate_parameters']

    device = init_run(parameters['seed'])

    print('Setting up evaluation')
    print(f'\t device: {device}')

    # We load the geometry object
    downsampling = evaluate_parameters['downsampling']
    geo = build_geometry(downsampling)
    geometry = geo.geometry

    # Now the models
    # 1) The pseudo inverse
    pseudo_inverse : torch.nn.Module = build_pseudo_inverse_from_params(
        train_parameters, geometry, device
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

    log_ignite_metrics(evaluator, test_dataloader, live, downsampling)

if __name__ == '__main__':
    evaluate_loop()
