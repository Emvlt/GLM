"""
Aggregates dvc exp results into mean +/- std PSNR/SSIM per config, across
whatever seeds/configs have been run (REVISION_PLAN.md P1 statistical-rigor
item 3 -- the new Table 1).

Reads `dvc exp show --json`, not the dvclive/ files directly: experiments
queued/run via queue_statistical_experiments.py only exist as dvc exp refs,
not as separate directories on disk, so `dvc exp show` is the only place
that correlates each run's params (seed, active_model, n_channels,
downsampling) with its metrics (Test PSNR/Test SSIM from
dvclive/evaluate/metrics.json).

Experiments whose metrics can't be read (e.g. missing from the local DVC
cache -- this currently affects the historical FBP_*/CNN_*/GLM_* sweep
experiments, since no DVC remote is configured) are skipped with a warning,
not silently dropped.
"""
import argparse
import csv
import json
import subprocess
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, List, Optional

METRICS_FILE = 'dvclive/evaluate/metrics.json'
DEFAULT_OUTPUT = 'results/aggregated_results.csv'

def load_experiments() -> List[Dict]:
    output = subprocess.run(
        ['dvc', 'exp', 'show', '--json', '--all-commits'],
        capture_output=True, text=True, check=True,
        ).stdout
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

def extract_record(exp: Dict) -> Optional[Dict]:
    name = exp.get('name')
    if not name:
        return None

    data = exp.get('data') or {}
    metric_entry = data.get('metrics', {}).get(METRICS_FILE, {})
    metrics = metric_entry.get('data')
    if metrics is None:
        print(f"Skipping '{name}': {METRICS_FILE} not readable ({metric_entry.get('error', {}).get('type', 'no data')})")
        return None

    params = data.get('params', {}).get('params.yaml', {}).get('data', {})
    seed = params.get('seed')
    pretrain_parameters = params.get('pretrain_parameters', {})
    active_model = pretrain_parameters.get('active_model')
    n_channels = pretrain_parameters.get('models', {}).get(active_model, {}).get('n_channels')
    downsampling = params.get('evaluate_parameters', {}).get('downsampling')

    if seed is None or active_model is None or n_channels is None:
        print(f"Skipping '{name}': missing seed/active_model/n_channels in params")
        return None

    if 'Test PSNR' not in metrics or 'Test SSIM' not in metrics:
        print(f"Skipping '{name}': {METRICS_FILE} missing 'Test PSNR'/'Test SSIM'")
        return None

    return {
        'name': name,
        'seed': seed,
        'active_model': active_model,
        'n_channels': n_channels,
        'downsampling': downsampling,
        'psnr': metrics['Test PSNR'],
        'ssim': metrics['Test SSIM'],
        }

def aggregate(records: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for record in records:
        key = (record['active_model'], record['n_channels'], record['downsampling'])
        groups[key].append(record)

    rows = []
    for (active_model, n_channels, downsampling), group in sorted(groups.items(), key=str):
        psnr_values = [r['psnr'] for r in group]
        ssim_values = [r['ssim'] for r in group]
        seeds = sorted({r['seed'] for r in group})
        rows.append({
            'active_model': active_model,
            'n_channels': n_channels,
            'downsampling': downsampling,
            'n_seeds': len(seeds),
            'seeds': seeds,
            'psnr_mean': mean(psnr_values),
            'psnr_std': stdev(psnr_values) if len(psnr_values) > 1 else None,
            'ssim_mean': mean(ssim_values),
            'ssim_std': stdev(ssim_values) if len(ssim_values) > 1 else None,
            })
    return rows

def format_std(value: Optional[float]) -> str:
    return f'{value:.4f}' if value is not None else 'n/a (1 seed)'

def print_table(rows: List[Dict]) -> None:
    header = f"{'model':<14}{'channels':<10}{'downsampling':<14}{'n_seeds':<9}{'PSNR mean':<12}{'PSNR std':<16}{'SSIM mean':<12}{'SSIM std':<16}"
    print(header)
    print('-' * len(header))
    for row in rows:
        print(
            f"{row['active_model']:<14}{row['n_channels']:<10}{str(row['downsampling']):<14}"
            f"{row['n_seeds']:<9}{row['psnr_mean']:<12.4f}{format_std(row['psnr_std']):<16}"
            f"{row['ssim_mean']:<12.4f}{format_std(row['ssim_std']):<16}"
            )

def write_csv(rows: List[Dict], path: str) -> None:
    import pathlib
    pathlib.Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'active_model', 'n_channels', 'downsampling', 'n_seeds', 'seeds',
            'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std',
            ])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'seeds': ','.join(str(s) for s in row['seeds'])})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='CSV output path')
    args = parser.parse_args()

    experiments = flatten_experiments(load_experiments())
    records = [r for r in (extract_record(exp) for exp in experiments) if r is not None]

    if not records:
        print('\nNo experiments with readable metrics found.')
        return

    rows = aggregate(records)
    print()
    print_table(rows)
    write_csv(rows, args.output)
    print(f'\nWrote {len(rows)} config(s) from {len(records)} experiment(s) to {args.output}')

if __name__ == '__main__':
    main()
