"""
Aggregates dvc exp results for the last round of the GLM ablation study
(REVISION_PLAN.md P1): the distance_mode x kernel_type x sigma sweep queued
by queue_sinogram_denoising_experiments.py's `ablation` family, re-run at
--input-mode mode2 --seeded, restricted to whichever of those seeded runs
have actually completed (seeds 270720261/62/63 as of writing; 270720264/65
from SEEDS haven't been run for this family yet).

Sibling to aggregate_sinogram_denoising_results.py, not a replacement: that
script's `ablation` family output (sinogram_denoising_ablation_mode2.csv)
mixes this seeded round together with the original single default-seed
ablation run (input_mode=mode2, no -seed suffix), giving n_seeds=4 with an
unmatched sample size across configs. This script only reads the seeded
round, so every config gets a clean, comparable n_seeds and a real std.

Works around the same dvclive/dvc-exp-queue quirk as its siblings: the
cached output of a completed evaluate_sinogram_denoising@<downsampling> run
is missing metrics.json (written by dvclive at process-exit, after dvc has
already hashed the stage's outputs under `dvc exp run --queue`). This reads
the last row of plots/metrics/Test PSNR|SSIM.tsv via dvc.api instead,
falling back to metrics.json first in case that gets fixed later.

Must be run with the project's own environment (dvc is a pyproject.toml
dependency), e.g. `uv run python src/glm/aggregate_ablation_study_results.py`.
"""
import argparse
import csv
import json
import pathlib
import re
import subprocess
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

from dvc.api import DVCFileSystem

DEFAULT_OUTPUT_DIR = 'results'
DEFAULT_DOWNSAMPLING_SWEEP = [1, 2, 4, 6, 8, 10]
PARAMS_FILE = 'sinogram_denoising_params.yaml'
RECORDS_FILENAME = 'ablation_study_records.json'
NAME_PREFIX = 'ablation-'

DEFAULT_INPUT_MODE = 'mode2'
DEFAULT_SEEDS: Tuple[int, ...] = (270720261, 270720262, 270720263)

GROUP_KEYS: Tuple[str, ...] = ('input_mode', 'distance_mode', 'kernel_type', 'sigma', 'downsampling')

# refs/exps/<2-hex shard>/<38-hex remainder>/<exp-name>; shard + remainder is
# the baseline commit the experiment was queued against.
EXP_REF_PATTERN = re.compile(r'^refs/exps/([0-9a-f]{2})/([0-9a-f]{38})/')

def experiment_baseline_revs() -> List[str]:
    """Baseline commits with at least one queued/run experiment attached.

    `dvc exp show --all-commits` walks every commit in the repo to find
    these, which is O(commits) and gets very slow as history grows. The
    baseline is encoded in the exp ref name itself, so read it straight from
    git and pass it via `--rev` instead -- same experiments, no full-history
    walk.
    """
    output = subprocess.run(
        ['git', 'for-each-ref', '--format=%(refname)', 'refs/exps'],
        capture_output=True, text=True, check=True,
        ).stdout
    revs = set()
    for line in output.splitlines():
        match = EXP_REF_PATTERN.match(line)
        if match:
            revs.add(match.group(1) + match.group(2))
    return sorted(revs)

def load_experiments() -> List[Dict]:
    revs = experiment_baseline_revs()
    cmd = ['dvc', 'exp', 'show', '--json', '--all-commits'] if not revs else (
        ['dvc', 'exp', 'show', '--json'] + [arg for rev in revs for arg in ('--rev', rev)]
        )
    output = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(output)

def flatten_experiments(entries: List[Dict]) -> List[Dict]:
    flat = []
    for entry in entries:
        if 'revs' in entry:
            flat += flatten_experiments(entry['revs'])
        else:
            flat.append(entry)
        if entry.get('experiments'):
            flat += flatten_experiments(entry['experiments'])
    return flat

_filesystems: Dict[str, DVCFileSystem] = {}

def filesystem_for(rev: str) -> DVCFileSystem:
    if rev not in _filesystems:
        _filesystems[rev] = DVCFileSystem('.', rev=None if rev == 'workspace' else rev)
    return _filesystems[rev]

def read_tsv_last_value(rev: str, path: str) -> Optional[float]:
    try:
        with filesystem_for(rev).open(path, 'r') as f:
            rows = [line for line in f.read().splitlines() if line.strip()]
        if len(rows) < 2:
            return None
        return float(rows[-1].split('\t')[1])
    except Exception:
        return None

def read_metrics(exp: Dict, downsampling: int) -> Optional[Dict[str, float]]:
    metrics_path = f'dvclive/evaluate_sinogram_denoising/{downsampling}/metrics.json'
    metric_entry = (exp.get('data') or {}).get('metrics', {}).get(metrics_path)
    metrics = (metric_entry or {}).get('data')
    if metrics is not None and 'Test PSNR' in metrics and 'Test SSIM' in metrics:
        return {'psnr': metrics['Test PSNR'], 'ssim': metrics['Test SSIM']}

    rev = exp.get('rev')
    base = f'dvclive/evaluate_sinogram_denoising/{downsampling}/plots/metrics'
    psnr = read_tsv_last_value(rev, f'{base}/Test PSNR.tsv')
    ssim = read_tsv_last_value(rev, f'{base}/Test SSIM.tsv')
    if psnr is None or ssim is None:
        return None
    return {'psnr': psnr, 'ssim': ssim}

def extract_records(exp: Dict, input_mode: str, seeds: Tuple[int, ...]) -> List[Dict]:
    name = exp.get('name')
    if not name or not name.startswith(NAME_PREFIX):
        return []

    data = exp.get('data') or {}
    params = data.get('params', {}).get(PARAMS_FILE, {}).get('data', {})
    seed = params.get('seed')
    if seed not in seeds:
        return []

    hyperparameters = params.get('hyperparameters', {})
    if hyperparameters.get('input_mode') != input_mode:
        return []

    model_parameters = params.get('model_parameters', {})
    active_model = model_parameters.get('active_model')
    model = model_parameters.get('models', {}).get(active_model, {})
    graph_kwargs = model.get('graph_kwargs', {})
    kernel_type = graph_kwargs.get('kernel_type')

    sweep = (
        data.get('params', {}).get('params.yaml', {}).get('data', {})
        .get('evaluate_parameters', {}).get('downsampling_sweep')
        ) or DEFAULT_DOWNSAMPLING_SWEEP

    common = {
        'name': name,
        'seed': seed,
        'input_mode': input_mode,
        'distance_mode': graph_kwargs.get('distance_mode'),
        'kernel_type': kernel_type,
        'sigma': graph_kwargs.get('sigma') if kernel_type == 'gaussian' else None,
        }

    records = []
    for downsampling in sweep:
        metrics = read_metrics(exp, downsampling)
        if metrics is None:
            print(f"Skipping '{name}' downsampling={downsampling}: no readable metrics.json or plots/metrics/Test PSNR|SSIM.tsv")
            continue
        records.append({**common, 'downsampling': downsampling, **metrics})

    if not records:
        print(f"Skipping '{name}': no readable downsampling metrics found")
    return records

def aggregate(records: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for record in records:
        groups[tuple(record[k] for k in GROUP_KEYS)].append(record)

    rows = []
    for key, group in sorted(groups.items(), key=str):
        psnr_values = [r['psnr'] for r in group]
        ssim_values = [r['ssim'] for r in group]
        seeds = sorted({r['seed'] for r in group})
        row = dict(zip(GROUP_KEYS, key))
        row.update({
            'n_seeds': len(seeds),
            'seeds': seeds,
            'psnr_mean': mean(psnr_values),
            'psnr_std': stdev(psnr_values) if len(psnr_values) > 1 else None,
            'ssim_mean': mean(ssim_values),
            'ssim_std': stdev(ssim_values) if len(ssim_values) > 1 else None,
            })
        rows.append(row)
    return rows

def format_std(value: Optional[float]) -> str:
    return f'{value:.4f}' if value is not None else 'n/a (1 seed)'

def print_table(rows: List[Dict]) -> None:
    col_width = 14
    header = ''.join(f'{key:<{col_width}}' for key in GROUP_KEYS)
    header += f"{'n_seeds':<9}{'PSNR mean':<12}{'PSNR std':<16}{'SSIM mean':<12}{'SSIM std':<16}"
    print(header)
    print('-' * len(header))
    for row in rows:
        line = ''.join(f'{str(row[key]):<{col_width}}' for key in GROUP_KEYS)
        line += f"{row['n_seeds']:<9}{row['psnr_mean']:<12.4f}{format_std(row['psnr_std']):<16}"
        line += f"{row['ssim_mean']:<12.4f}{format_std(row['ssim_std']):<16}"
        print(line)

def write_csv(rows: List[Dict], path: str) -> None:
    fieldnames = [*GROUP_KEYS, 'n_seeds', 'seeds', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'seeds': ','.join(str(s) for s in row['seeds'])})

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='directory for CSV output')
    parser.add_argument('--input-mode', default=DEFAULT_INPUT_MODE, help='hyperparameters.input_mode to filter on')
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS),
        help='seeds to include, i.e. the round to extract (default: the completed seeds of the last round)',
        )
    args = parser.parse_args()
    seeds = tuple(args.seeds)

    experiments = flatten_experiments(load_experiments())
    records = []
    for exp in experiments:
        records += extract_records(exp, args.input_mode, seeds)

    if not records:
        print(f'\nNo ablation experiments found for input_mode={args.input_mode}, seeds={seeds}.')
        return

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    records_path = pathlib.Path(args.output_dir) / RECORDS_FILENAME
    records_path.write_text(json.dumps(records, indent=2))
    print(f'Wrote {len(records)} raw per-seed record(s) to {records_path}')

    rows = aggregate(records)
    print(f'\n=== ablation_study ({args.input_mode}, seeds={seeds}) ===')
    print_table(rows)
    output_path = pathlib.Path(args.output_dir) / 'ablation_study_results.csv'
    write_csv(rows, str(output_path))
    print(f'Wrote {len(rows)} config(s) from {len(records)} experiment-downsampling record(s) to {output_path}')

if __name__ == '__main__':
    main()
