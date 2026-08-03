import argparse
import json
from statistics import mean, median
from time import perf_counter
from typing import Dict, List, Optional

import torch

from glm.config import init_run, load_params, SINOGRAM_DENOISING_PARAMS_PATH
from glm.models.utils import (
    N_PIXELS, build_batched_graph, build_geometry, forward_sinogram_model,
    load_graph, load_model,
    )

try:
    from torch.utils.flop_counter import FlopCounterMode
    _HAS_FLOP_COUNTER = True
except ImportError:
    _HAS_FLOP_COUNTER = False

DEFAULT_N_WARMUP = 10
DEFAULT_N_RUNS = 50

def _time_calls(fn, n_warmup: int, n_runs: int, device: torch.device) -> Dict[str, float]:
    for _ in range(n_warmup):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    timings_ms = []
    for _ in range(n_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = perf_counter()
        fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        timings_ms.append(1e3 * (perf_counter() - start))

    return {
        'mean_ms': mean(timings_ms),
        'median_ms': median(timings_ms),
        'min_ms': min(timings_ms),
        'max_ms': max(timings_ms),
    }

def _count_flops(fn) -> Optional[int]:
    if not _HAS_FLOP_COUNTER:
        return None
    try:
        with FlopCounterMode(display=False) as flop_counter:
            fn()
        return flop_counter.get_total_flops()
    except Exception as error:  # pragma: no cover - defensive, some ops may not be traceable
        print(f'Warning: FLOP counting failed ({error!r}), skipping.')
        return None

def benchmark_model(
        model_name: str,
        model_parameters: Dict,
        geometry_config,
        batch_size: int,
        device: torch.device,
        n_warmup: int = DEFAULT_N_WARMUP,
        n_runs: int = DEFAULT_N_RUNS,
        ) -> Dict:
    angles_indices = geometry_config.angles_indices
    n_measurements = geometry_config.n_measurements
    geometry = geometry_config.geometry

    model = load_model(model_name, model_parameters).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    graph_kwargs = model_parameters.get('graph_kwargs')
    graph = load_graph(model_name, geometry, **(graph_kwargs or {}))
    graphs = build_batched_graph(graph, batch_size, device)

    input_sinogram = torch.randn(batch_size, 1, n_measurements, N_PIXELS, device=device)
    target_sinogram = torch.randn(batch_size, 1, n_measurements, N_PIXELS, device=device)

    def run_forward():
        return forward_sinogram_model(
            model, input_sinogram,
            batch_size=batch_size,
            angles_indices=angles_indices,
            n_measurements=n_measurements,
            graph=graph,
            graphs=graphs,
            reshape_to_tomo=True,
            )

    loss_function = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

    def run_train_step():
        optimiser.zero_grad()
        _, infered_sinogram = run_forward()
        loss = loss_function(infered_sinogram, target_sinogram)
        loss.backward()
        optimiser.step()

    def run_inference():
        with torch.no_grad():
            run_forward()

    model.eval()
    inference_time_ms = _time_calls(run_inference, n_warmup, n_runs, device)
    inference_flops = _count_flops(run_inference)

    model.train()
    training_step_time_ms = _time_calls(run_train_step, n_warmup, n_runs, device)
    training_step_flops = _count_flops(run_train_step)

    return {
        'model': model_name,
        'batch_size': batch_size,
        'n_measurements': n_measurements,
        'n_params': n_params,
        'inference_time_ms': inference_time_ms,
        'inference_flops': inference_flops,
        'training_step_time_ms': training_step_time_ms,
        'training_step_flops': training_step_flops,
    }

def main(
        params_path: str = SINOGRAM_DENOISING_PARAMS_PATH,
        model_names: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        n_warmup: int = DEFAULT_N_WARMUP,
        n_runs: int = DEFAULT_N_RUNS,
        output_path: Optional[str] = 'results/benchmark_sinogram_denoising_models.json',
        ) -> List[Dict]:
    parameters = load_params(params_path)
    hyperparameters = parameters['hyperparameters']
    model_parameters = parameters['model_parameters']

    device = init_run(parameters['seed'])
    geometry_config = build_geometry(hyperparameters['downsampling'])

    if model_names is None:
        model_names = list(model_parameters['models'].keys())
    if batch_size is None:
        batch_size = hyperparameters['batch_size']

    if not _HAS_FLOP_COUNTER:
        print('Warning: torch.utils.flop_counter is unavailable (requires torch>=2.1); FLOPs will be reported as null.')

    results = []
    for model_name in model_names:
        print(f'Benchmarking {model_name} (batch_size={batch_size})...')
        result = benchmark_model(
            model_name,
            model_parameters['models'][model_name],
            geometry_config,
            batch_size,
            device,
            n_warmup=n_warmup,
            n_runs=n_runs,
            )
        results.append(result)
        print(json.dumps(result, indent=4))

    if output_path is not None:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f'Saved benchmark results to {output_path}')

    return results

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--params', type=str, default=SINOGRAM_DENOISING_PARAMS_PATH,
        help='Path to the yaml file holding the sinogram denoising experiment parameters.'
        )
    arg_parser.add_argument(
        '--models', type=str, nargs='+', default=None,
        help='Which models (from model_parameters.models) to benchmark. Defaults to all of them.'
        )
    arg_parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Overrides hyperparameters.batch_size from the params yaml.'
        )
    arg_parser.add_argument('--n-warmup', type=int, default=DEFAULT_N_WARMUP)
    arg_parser.add_argument('--n-runs', type=int, default=DEFAULT_N_RUNS)
    arg_parser.add_argument(
        '--output', type=str, default='results/benchmark_sinogram_denoising_models.json',
        help='Optional path to dump the benchmark results as JSON.'
        )
    args = arg_parser.parse_args()
    main(args.params, args.models, args.batch_size, args.n_warmup, args.n_runs, args.output)
