"""
Exports the last round of the GLM ablation study (see
aggregate_ablation_study_results.py) as paper-ready LaTeX tables: a
distance_mode x kernel_type/sigma grid across the downsampling sweep, one
table per metric (PSNR, SSIM).

Reads results/ablation_study_results.csv, whose psnr_mean/ssim_mean columns
are already averaged across the round's 3 seeds -- this script just lays
those out as a grid, no further seed handling needed. Columns are the
downsampling sweep rather than a single native-resolution slice (unlike
export_depth_study_latex.py) because REVISION_PLAN.md's P1 ablation table
specifically flags whether the Gaussian kernel's bandwidth sigma needs to
scale with angular spacing under downsampling -- that's only visible if
sigma variants are compared across the sweep, not at downsampling=1 alone.

Run aggregate_ablation_study_results.py first to (re)generate the CSV --
this script has no dvc dependency of its own.
"""
import argparse
import csv
import pathlib
from typing import Dict, Optional, Tuple

DEFAULT_OUTPUT_DIR = 'results/latex'
DEFAULT_RESULTS_DIR = 'results'
RESULTS_FILENAME = 'ablation_study_results.csv'

DOWNSAMPLINGS = (1, 2, 4, 6, 8, 10)
DISTANCE_MODES = (('surface', 'Surface (geodesic)'), ('euclidean', 'Euclidean'))
KERNEL_CONFIGS = (('gaussian', 0.5), ('gaussian', 1.0), ('gaussian', 2.0), ('inverse', None))
METRICS = (('psnr_mean', 'psnr', 'PSNR (dB)', '.2f'), ('ssim_mean', 'ssim', 'SSIM', '.3f'))

GridKey = Tuple[str, str, Optional[float]]


def kernel_label(kernel_type: str, sigma: Optional[float]) -> str:
    return f'Gaussian ($\\sigma={sigma}$)' if kernel_type == 'gaussian' else 'Inverse'


def parse_sigma(raw: str) -> Optional[float]:
    return float(raw) if raw not in ('', 'None') else None


def read_ablation_grid(path: pathlib.Path) -> Dict[GridKey, Dict[int, Dict[str, float]]]:
    """{(distance_mode, kernel_type, sigma): {downsampling: {'psnr_mean': .., 'ssim_mean': .., 'n_seeds': ..}}}."""
    grid: Dict[GridKey, Dict[int, Dict[str, float]]] = {}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            key = (row['distance_mode'], row['kernel_type'], parse_sigma(row['sigma']))
            downsampling = int(row['downsampling'])
            grid.setdefault(key, {})[downsampling] = {
                'psnr_mean': float(row['psnr_mean']),
                'ssim_mean': float(row['ssim_mean']),
                'n_seeds': int(row['n_seeds']),
                }
    return grid


def write_table(
    grid: Dict[GridKey, Dict[int, Dict[str, float]]], input_mode: str,
    metric_key: str, metric_label: str, fmt: str, output_path: pathlib.Path,
    ) -> None:
    n_seeds_seen = {
        cell['n_seeds']
        for by_downsampling in grid.values()
        for cell in by_downsampling.values()
        }
    seeds_note = f'{min(n_seeds_seen)}' if len(n_seeds_seen) == 1 else f'{min(n_seeds_seen)}--{max(n_seeds_seen)}'

    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\begin{tabular}{ll' + 'c' * len(DOWNSAMPLINGS) + '}')
    lines.append('\\toprule')
    header = 'Distance & Kernel & ' + ' & '.join(
        (f'${d}$ (native)' if d == 1 else f'${d}$') for d in DOWNSAMPLINGS
        )
    lines.append(header + ' \\\\')
    lines.append('\\midrule')
    for distance_mode, distance_label in DISTANCE_MODES:
        for row_index, (kernel_type, sigma) in enumerate(KERNEL_CONFIGS):
            by_downsampling = grid.get((distance_mode, kernel_type, sigma), {})
            cells = []
            for d in DOWNSAMPLINGS:
                value = by_downsampling.get(d, {}).get(metric_key)
                cells.append(f'{value:{fmt}}' if value is not None else '--')
            label = f'\\multirow{{{len(KERNEL_CONFIGS)}}}{{*}}{{{distance_label}}}' if row_index == 0 else ''
            lines.append(f'{label} & {kernel_label(kernel_type, sigma)} & ' + ' & '.join(cells) + ' \\\\')
        lines.append('\\midrule')
    lines[-1] = '\\bottomrule'
    lines.append('\\end{tabular}')
    lines.append(
        f'\\caption{{{metric_label} vs.\\ GLM edge distance metric and heat-kernel type across the angular '
        f'downsampling sweep, mode2, mean over {seeds_note} seeds.}}'
        )
    lines.append(f'\\label{{tab:ablation-study-{input_mode}-{metric_key.split("_")[0]}}}')
    lines.append('\\end{table}')
    output_path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='directory for LaTeX tables')
    parser.add_argument(
        '--results-dir', default=DEFAULT_RESULTS_DIR,
        help=f'directory holding {RESULTS_FILENAME}, written by aggregate_ablation_study_results.py',
        )
    args = parser.parse_args()

    results_path = pathlib.Path(args.results_dir) / RESULTS_FILENAME
    if not results_path.exists():
        print(f'{results_path} not found -- run aggregate_ablation_study_results.py first.')
        return

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    grid = read_ablation_grid(results_path)
    if not grid:
        print(f'No rows in {results_path}.')
        return

    input_mode = 'mode2'
    n_configs = len(DISTANCE_MODES) * len(KERNEL_CONFIGS)
    master_lines = []
    for metric_key, metric_stem, metric_label, fmt in METRICS:
        table_path = output_dir / f'ablation_study_{metric_stem}_table.tex'
        write_table(grid, input_mode, metric_key, metric_label, fmt, table_path)
        master_lines.append(f'\\input{{{table_path.name}}}')
        master_lines.append('')
    print(f'Wrote ablation-study tables ({len(grid)}/{n_configs} configs found)')

    master_path = output_dir / 'ablation_study.tex'
    master_path.write_text('\n'.join(master_lines))
    print(f'Wrote {master_path} (includes both ablation-study tables)')


if __name__ == '__main__':
    main()
