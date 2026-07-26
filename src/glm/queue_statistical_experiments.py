"""
Queues the REVISION_PLAN.md P1 statistical-rigor sweep: 5 seeds x the
existing {sinogram_CNN, GLM} x {16, 24 channels} matrix (20 experiments).

The `evaluate` stage in dvc.yaml is now a `foreach` over
evaluate_parameters.downsampling_sweep, so each of these 20 queued
experiments automatically reproduces evaluate@1..evaluate@10 (the full
downsampling sweep) once training completes -- no separate downsampling
axis needed here.

This only queues experiments (`dvc exp run --queue`) -- nothing executes
until you run `dvc queue start` / `dvc exp run --run-all` yourself. With a
single GPU, run them one at a time (no `-j` / `-j 1`) to avoid contention.

Only covers what's fully unblocked today: the sinogram-model seed/channel
matrix (each with the full downsampling sweep). NOT included (see
REVISION_PLAN.md for why):
  - FBP-only baseline: evaluate_FBP isn't a dvc.yaml stage right now.
  - aggr / GCN-flag ablations: wired in params.yaml, just not queued here.
  - depth, message-passing on/off: need a code change in models/utils.py / gnn.py.
  - distance metric, connectivity scheme, sigma bandwidth: blocked on the
    odl fork (see the sigma finding in REVISION_PLAN.md's GLM ablation section).
"""
import argparse
import subprocess
from typing import Dict, List

SEEDS: List[int] = [1, 2, 3, 4, 5]
SINOGRAM_MODEL_MATRIX: List[Dict] = [
    {'active_model': 'sinogram_CNN', 'n_channels': 16},
    {'active_model': 'sinogram_CNN', 'n_channels': 24},
    {'active_model': 'GLM', 'n_channels': 16},
    {'active_model': 'GLM', 'n_channels': 24},
]

def build_overrides(seed: int, active_model: str, n_channels: int) -> List[str]:
    return [
        f'seed={seed}',
        f'pretrain_parameters.active_model={active_model}',
        f'pretrain_parameters.models.{active_model}.n_channels={n_channels}',
    ]

def build_name(seed: int, active_model: str, n_channels: int) -> str:
    return f'seed{seed}-{active_model}{n_channels}'

def queue_experiment(seed: int, active_model: str, n_channels: int, dry_run: bool) -> None:
    name = build_name(seed, active_model, n_channels)
    command = ['dvc', 'exp', 'run', '--queue', '-n', name]
    for override in build_overrides(seed, active_model, n_channels):
        command += ['-S', override]

    print(' '.join(command))
    if not dry_run:
        subprocess.run(command, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dry-run', action='store_true',
        help='print the dvc commands without queuing anything',
        )
    args = parser.parse_args()

    for seed in SEEDS:
        for config in SINOGRAM_MODEL_MATRIX:
            queue_experiment(seed, config['active_model'], config['n_channels'], args.dry_run)

    n_queued = len(SEEDS) * len(SINOGRAM_MODEL_MATRIX)
    if args.dry_run:
        print(f'\n{n_queued} experiments would be queued (each reproducing the full downsampling sweep).')
    else:
        print(f'\n{n_queued} experiments queued (each reproducing the full downsampling sweep). Nothing has run yet.')
        print('Single GPU: run them one at a time with `dvc queue start` or `dvc exp run --run-all` (no -j).')
        print('Inspect the queue with `dvc queue status`; drop it with `dvc queue remove --all`.')

if __name__ == '__main__':
    main()
