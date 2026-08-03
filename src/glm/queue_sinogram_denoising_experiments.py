"""
Queues sinogram-denoising experiments via `dvc exp run --queue`, following the
same convention as queue_statistical_experiments.py but targeting the
sinogram_denoising_params.yaml-driven `sinogram_denoising` / `evaluate_sinogram_denoising`
dvc.yaml stages instead of the combined pretrain/train pipeline.

Three families, corresponding to REVISION_PLAN.md's P2 mode-generalization item
and P1 GLM ablation study table:

  mode      -- statistical rigor for mode1/mode2/mode3 -> mode2 denoising:
               {mode1, mode2, mode3} input modes x {sinogram_CNN, GLM} x
               {16, 24} channels x 5 seeds (60 experiments).
  ablation  -- GLM distance metric / kernel ablation:
               distance_mode {surface, euclidean} x kernel_type {gaussian, inverse},
               with kernel_type=gaussian further swept over sigma {0.5, 1.0, 2.0}
               (8 experiments; sigma is meaningless for kernel_type=inverse so it
               isn't swept there).
  depth     -- connectivity vs number of GLM modules grid:
               connectivity {1,2,3} x n_modules {1,2,3,4} (12 experiments).

This only queues experiments (`dvc exp run --queue`) -- nothing executes until
you run `dvc queue start` / `dvc exp run --run-all` yourself. With a single
GPU, run them one at a time (no `-j` / `-j 1`) to avoid contention.
"""
import argparse
import subprocess
from typing import Dict, List

PARAMS_FILE = 'sinogram_denoising_params.yaml'
TARGET = 'evaluate_sinogram_denoising'  # depends on `sinogram_denoising`, pulls it in automatically

SEEDS: List[int] = [270720261, 270720262, 270720263, 270720264, 270720265]
MODE_PAIRS: List[tuple] = [('mode1', 'mode2'), ('mode2', 'mode2'), ('mode3', 'mode2')]
ACTIVE_MODELS: List[str] = ['sinogram_CNN', 'GLM']
N_CHANNELS: List[int] = [16, 24]

DISTANCE_MODES: List[str] = ['surface', 'euclidean']
KERNEL_TYPES: List[str] = ['gaussian', 'inverse']
SIGMAS: List[float] = [0.5, 1.0, 2.0]

CONNECTIVITIES: List[int] = [1, 2, 3]
N_MODULES: List[int] = [1, 2, 3, 4]


def mode_experiments() -> List[Dict]:
    return [
        {
            'name': f'mode-{input_mode}-{target_mode}-{active_model}{n_channels}-seed{seed}',
            'overrides': {
                'seed': seed,
                'hyperparameters.input_mode': input_mode,
                'hyperparameters.target_mode': target_mode,
                'model_parameters.active_model': active_model,
                f'model_parameters.models.{active_model}.n_channels': n_channels,
                },
            }
        for input_mode, target_mode in MODE_PAIRS
        for active_model in ACTIVE_MODELS
        for n_channels in N_CHANNELS
        for seed in SEEDS
        ]


def _seed_experiments(experiments: List[Dict], seeds: List[int] = None) -> List[Dict]:
    """Expands each experiment into one variant per seed, seed suffixed onto
    the name and added to its overrides. `seeds=None` (the default) leaves
    experiments untouched -- ablation/depth are single runs at the params
    file's default seed unless a seed sweep is explicitly requested.
    """
    if not seeds:
        return experiments
    return [
        {'name': f'{experiment["name"]}-seed{seed}', 'overrides': {**experiment['overrides'], 'seed': seed}}
        for experiment in experiments
        for seed in seeds
        ]


def ablation_experiments(input_mode: str = None, seeds: List[int] = None) -> List[Dict]:
    name_suffix = f'-{input_mode}' if input_mode else ''
    mode_overrides = {'hyperparameters.input_mode': input_mode} if input_mode else {}
    experiments = []
    for distance_mode in DISTANCE_MODES:
        for kernel_type in KERNEL_TYPES:
            base_overrides = {
                'model_parameters.active_model': 'GLM',
                'model_parameters.models.GLM.graph_kwargs.distance_mode': distance_mode,
                'model_parameters.models.GLM.graph_kwargs.kernel_type': kernel_type,
                **mode_overrides,
                }
            if kernel_type == 'gaussian':
                for sigma in SIGMAS:
                    experiments.append({
                        'name': f'ablation-{distance_mode}-{kernel_type}-sigma{sigma}{name_suffix}',
                        'overrides': {**base_overrides, 'model_parameters.models.GLM.graph_kwargs.sigma': sigma},
                        })
            else:
                experiments.append({
                    'name': f'ablation-{distance_mode}-{kernel_type}{name_suffix}',
                    'overrides': base_overrides,
                    })
    return _seed_experiments(experiments, seeds)


def depth_experiments(input_mode: str = None, batch_size: int = None, seeds: List[int] = None) -> List[Dict]:
    name_suffix = f'-{input_mode}' if input_mode else ''
    name_suffix += f'-bs{batch_size}' if batch_size else ''
    mode_overrides = {'hyperparameters.input_mode': input_mode} if input_mode else {}
    batch_size_overrides = {'hyperparameters.batch_size': batch_size} if batch_size else {}
    experiments = [
        {
            'name': f'depth-connectivity{connectivity}-nmodules{n_modules}{name_suffix}',
            'overrides': {
                'model_parameters.active_model': 'GLM',
                'model_parameters.models.GLM.graph_kwargs.connectivity': connectivity,
                'model_parameters.models.GLM.n_modules': n_modules,
                **mode_overrides,
                **batch_size_overrides,
                },
            }
        for connectivity in CONNECTIVITIES
        for n_modules in N_MODULES
        ]
    return _seed_experiments(experiments, seeds)


FAMILIES = {
    'mode': mode_experiments,
    'ablation': ablation_experiments,
    'depth': depth_experiments,
    }


def queue_experiment(name: str, overrides: Dict, dry_run: bool) -> None:
    command = ['dvc', 'exp', 'run', '--queue', '-n', name]
    for key, value in overrides.items():
        command += ['-S', f'{PARAMS_FILE}:{key}={value}']
    command += [TARGET]

    print(' '.join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--family', choices=[*FAMILIES.keys(), 'all'], default='all',
        help='which experiment family to queue (default: all)',
        )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='print the dvc commands without queuing anything',
        )
    parser.add_argument(
        '--input-mode', default=None,
        help='override hyperparameters.input_mode for the depth/ablation families (default: mode1, from the params file)',
        )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='override hyperparameters.batch_size for the depth family (default: from the params file)',
        )
    parser.add_argument(
        '--seeded', action='store_true',
        help='sweep the ablation/depth families over the same 5 seeds as the mode family (SEEDS), '
             'instead of a single run at the params-file default seed',
        )
    args = parser.parse_args()

    families = FAMILIES.keys() if args.family == 'all' else [args.family]
    seeds = SEEDS if args.seeded else None

    n_queued = 0
    for family in families:
        if family == 'depth':
            experiments = depth_experiments(input_mode=args.input_mode, batch_size=args.batch_size, seeds=seeds)
        elif family == 'ablation':
            experiments = ablation_experiments(input_mode=args.input_mode, seeds=seeds)
        else:
            experiments = FAMILIES[family]()
        print(f'\n--- {family} ({len(experiments)} experiments) ---')
        for experiment in experiments:
            queue_experiment(experiment['name'], experiment['overrides'], args.dry_run)
        n_queued += len(experiments)

    if args.dry_run:
        print(f'\n{n_queued} experiments would be queued.')
    else:
        print(f'\n{n_queued} experiments queued. Nothing has run yet.')
        print('Single GPU: run them one at a time with `dvc queue start` or `dvc exp run --run-all` (no -j).')
        print('Inspect the queue with `dvc queue status`; drop it with `dvc queue remove --all`.')


if __name__ == '__main__':
    main()
