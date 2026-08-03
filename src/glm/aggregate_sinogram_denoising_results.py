"""
Aggregates dvc exp results for the sinogram-denoising sweep queued by
queue_sinogram_denoising_experiments.py (REVISION_PLAN.md P2
mode-generalization item and P1 GLM ablation study), across whatever
experiments have been run so far.

This is a sibling to aggregate_results.py, not a replacement: that script
reads the `evaluate` (end-to-end reconstruction) stage, driven by
params.yaml. Nothing queued via queue_sinogram_denoising_experiments.py ever
touches that stage -- it only targets `evaluate_sinogram_denoising` (which
pulls in `sinogram_denoising`), driven by sinogram_denoising_params.yaml. So
this reads that stage instead, and correlates against its own params.

Also works around a second issue: as of writing, the cached output of every
completed evaluate_sinogram_denoising@<downsampling> run is missing
`metrics.json` (the file dvc.yaml's `metrics:` list -- and `dvc exp show`
--json -- exposes). Only `plots/metrics/Test PSNR.tsv` and `Test SSIM.tsv`
made it into the cache; metrics.json looks like it's written by dvclive at
process-exit (see log_ignite_metrics in config.py) after DVC has already
hashed the stage's outputs when run under `dvc exp run --queue`, while the
per-step .tsv rows are flushed synchronously during the run and always make
it in. Both files carry the same single value (log_ignite_metrics logs each
metric exactly once), so this reads the last .tsv row via dvc.api instead,
falling back to metrics.json first in case that gets fixed later.

Splits output into the three families queue_sinogram_denoising_experiments.py
queues -- mode / ablation / depth -- since they vary different parameters.
Only `mode` is seeded (5 seeds); ablation/depth configs are single runs, so
their n_seeds will always be 1.

ablation/depth were originally only ever queued at the default input_mode
(mode1, from sinogram_denoising_params.yaml). queue_sinogram_denoising_experiments.py
now also supports queuing them at --input-mode mode2, so a given (config,
downsampling) can have both a mode1 and a mode2 run. input_mode is part of
GROUP_KEYS for these two families so they never get merged into one group,
and their output is further split into one CSV per input_mode (e.g.
sinogram_denoising_ablation_mode1.csv / _mode2.csv) so the recent mode2 runs
stay visibly separate from the earlier mode1 ones rather than sitting
side-by-side in an undifferentiated table.

Must be run with the project's own environment (dvc is a pyproject.toml
dependency), e.g. `uv run python src/glm/aggregate_sinogram_denoising_results.py`.
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
RECORDS_FILENAME = 'sinogram_denoising_records.json'

# refs/exps/<2-hex shard>/<38-hex remainder>/<exp-name>; shard + remainder is
# the baseline commit the experiment was queued against.
EXP_REF_PATTERN = re.compile(r'^refs/exps/([0-9a-f]{2})/([0-9a-f]{38})/')

FAMILIES = ('mode', 'ablation', 'depth')
GROUP_KEYS = {
    'mode': ('input_mode', 'target_mode', 'active_model', 'n_channels', 'downsampling'),
    'ablation': ('input_mode', 'distance_mode', 'kernel_type', 'sigma', 'downsampling'),
    'depth': ('input_mode', 'connectivity', 'n_modules', 'downsampling'),
    }
# ablation/depth get one CSV per input_mode instead of one combined table --
# see module docstring. `mode` stays combined since comparing input_modes
# against each other is the point of that family.
SPLIT_BY_INPUT_MODE = ('ablation', 'depth')

def experiment_baseline_revs() -> List[str]:
    """Baseline commits with at least one queued/run experiment attached.

    `dvc exp show --all-commits` walks every commit in the repo to find
    these, which is O(commits) and gets very slow as history grows (8m+
    here, against 199 commits for ~90 experiment refs). The baseline is
    encoded in the exp ref name itself, so read it straight from git and
    pass it via `--rev` instead -- same experiments, no full-history walk.
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

def classify(name: str) -> Optional[str]:
    for family in FAMILIES:
        if name.startswith(family + '-'):
            return family
    return None

def extract_records(exp: Dict) -> List[Dict]:
    name = exp.get('name')
    family = classify(name) if name else None
    if family is None:
        return []

    data = exp.get('data') or {}
    params = data.get('params', {}).get(PARAMS_FILE, {}).get('data', {})
    seed = params.get('seed')
    hyperparameters = params.get('hyperparameters', {})
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
        'family': family,
        'seed': seed,
        'input_mode': hyperparameters.get('input_mode'),
        'target_mode': hyperparameters.get('target_mode'),
        'active_model': active_model,
        'n_channels': model.get('n_channels'),
        'distance_mode': graph_kwargs.get('distance_mode'),
        'kernel_type': kernel_type,
        'sigma': graph_kwargs.get('sigma') if kernel_type == 'gaussian' else None,
        'connectivity': graph_kwargs.get('connectivity'),
        'n_modules': model.get('n_modules'),
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

def aggregate(records: List[Dict], family: str) -> List[Dict]:
    keys = GROUP_KEYS[family]
    groups = defaultdict(list)
    for record in records:
        groups[tuple(record[k] for k in keys)].append(record)

    rows = []
    for key, group in sorted(groups.items(), key=str):
        psnr_values = [r['psnr'] for r in group]
        ssim_values = [r['ssim'] for r in group]
        seeds = sorted({r['seed'] for r in group})
        row = dict(zip(keys, key))
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

def print_table(rows: List[Dict], keys: Tuple[str, ...]) -> None:
    col_width = 14
    header = ''.join(f'{key:<{col_width}}' for key in keys)
    header += f"{'n_seeds':<9}{'PSNR mean':<12}{'PSNR std':<16}{'SSIM mean':<12}{'SSIM std':<16}"
    print(header)
    print('-' * len(header))
    for row in rows:
        line = ''.join(f'{str(row[key]):<{col_width}}' for key in keys)
        line += f"{row['n_seeds']:<9}{row['psnr_mean']:<12.4f}{format_std(row['psnr_std']):<16}"
        line += f"{row['ssim_mean']:<12.4f}{format_std(row['ssim_std']):<16}"
        print(line)

def write_csv(rows: List[Dict], keys: Tuple[str, ...], path: str) -> None:
    fieldnames = [*keys, 'n_seeds', 'seeds', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'seeds': ','.join(str(s) for s in row['seeds'])})

def emit(rows: List[Dict], keys: Tuple[str, ...], n_records: int, label: str, filename_stem: str, output_dir: str) -> None:
    print(f'\n=== {label} ===')
    print_table(rows, keys)
    output_path = f'{output_dir}/{filename_stem}.csv'
    write_csv(rows, keys, output_path)
    print(f'Wrote {len(rows)} config(s) from {n_records} experiment-downsampling record(s) to {output_path}')

def load_csv_rows(path: pathlib.Path, keys: Tuple[str, ...]) -> List[Dict]:
    with open(path, newline='') as f:
        rows = []
        for row in csv.DictReader(f):
            parsed = {key: row[key] for key in keys}
            parsed['n_seeds'] = int(row['n_seeds'])
            parsed['seeds'] = row['seeds'].split(',') if row['seeds'] else []
            parsed['psnr_mean'] = float(row['psnr_mean'])
            parsed['ssim_mean'] = float(row['ssim_mean'])
            parsed['psnr_std'] = float(row['psnr_std']) if row['psnr_std'] not in ('', 'None') else None
            parsed['ssim_std'] = float(row['ssim_std']) if row['ssim_std'] not in ('', 'None') else None
            rows.append(parsed)
    return rows

def show_existing(output_dir: str) -> bool:
    """Prints whatever per-family result CSVs already exist in output_dir,
    without touching dvc/git -- skips the potentially slow experiment walk
    in load_experiments(), for re-inspecting the last aggregation run.
    Returns True if at least one CSV was found and printed.
    """
    output_path = pathlib.Path(output_dir)
    found_any = False
    for family in FAMILIES:
        keys = GROUP_KEYS[family]
        if family not in SPLIT_BY_INPUT_MODE:
            candidates = [(family, output_path / f'sinogram_denoising_{family}.csv')]
        else:
            candidates = [
                (f'{family} ({path.stem.rsplit("_", 1)[-1]})', path)
                for path in sorted(output_path.glob(f'sinogram_denoising_{family}_*.csv'))
                ]
        for label, path in candidates:
            if not path.exists():
                continue
            found_any = True
            print(f'\n=== {label} (from {path}) ===')
            print_table(load_csv_rows(path, keys), keys)
    return found_any

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='directory for per-family CSV output')
    parser.add_argument(
        '--show', action='store_true',
        help='print whatever result CSVs already exist in --output-dir and exit, '
             'skipping the dvc exp walk entirely',
        )
    args = parser.parse_args()

    if args.show:
        if not show_existing(args.output_dir):
            print(f'No result CSVs found in {args.output_dir}. Run without --show first to generate them.')
        return

    experiments = flatten_experiments(load_experiments())
    records_by_family = defaultdict(list)
    for exp in experiments:
        for record in extract_records(exp):
            records_by_family[record['family']].append(record)

    if not any(records_by_family.values()):
        print('\nNo experiments with readable metrics found.')
        return

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    all_records = [record for records in records_by_family.values() for record in records]
    records_path = pathlib.Path(args.output_dir) / RECORDS_FILENAME
    records_path.write_text(json.dumps(all_records, indent=2))
    print(f'Wrote {len(all_records)} raw per-seed record(s) to {records_path}')

    for family in FAMILIES:
        records = records_by_family.get(family, [])
        if not records:
            print(f'\nNo readable records for family={family}')
            continue
        rows = aggregate(records, family)
        keys = GROUP_KEYS[family]

        if family not in SPLIT_BY_INPUT_MODE:
            emit(rows, keys, len(records), family, f'sinogram_denoising_{family}', args.output_dir)
            continue

        for input_mode in sorted({row['input_mode'] for row in rows}):
            subset_rows = [row for row in rows if row['input_mode'] == input_mode]
            subset_n = sum(1 for r in records if r['input_mode'] == input_mode)
            emit(
                subset_rows, keys, subset_n, f'{family} ({input_mode})',
                f'sinogram_denoising_{family}_{input_mode}', args.output_dir,
                )

if __name__ == '__main__':
    main()
