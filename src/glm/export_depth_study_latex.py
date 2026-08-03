"""
Exports the sinogram-denoising `depth` family (see
aggregate_sinogram_denoising_results.py) as paper-ready LaTeX tables: a
connectivity x n_modules grid at native resolution (downsampling=1), one
table per (input_mode, metric).

depth configs are single runs (no seeds), so there's none of the
median/scatter/divergence-marking complexity export_sinogram_denoising_latex.py
needs for the seeded `mode` family -- this is a straight read of the
per-input-mode CSVs (results/sinogram_denoising_depth_<input_mode>.csv) that
aggregate_sinogram_denoising_results.py writes, filtered to downsampling=1.
Native resolution is what the connectivity/depth grid is about; robustness
to angular undersampling is a separate question already covered by the
`mode` family's own export.

Run aggregate_sinogram_denoising_results.py first to (re)generate those
CSVs -- this script has no dvc dependency of its own.
"""
import argparse
import csv
import pathlib
from typing import Dict, Tuple

DEFAULT_OUTPUT_DIR = 'results/latex'
DEFAULT_RESULTS_DIR = 'results'
NATIVE_DOWNSAMPLING = 1
CONNECTIVITIES = (1, 2, 3)
N_MODULES = (1, 2, 3, 4)
METRICS = (('psnr_mean', 'psnr', 'PSNR (dB)', '.2f'), ('ssim_mean', 'ssim', 'SSIM', '.3f'))


def read_depth_grid(path: pathlib.Path) -> Dict[Tuple[int, int], Dict[str, float]]:
    """{(connectivity, n_modules): {'psnr_mean': .., 'ssim_mean': ..}} at downsampling=1."""
    grid = {}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            if int(row['downsampling']) != NATIVE_DOWNSAMPLING:
                continue
            key = (int(row['connectivity']), int(row['n_modules']))
            grid[key] = {'psnr_mean': float(row['psnr_mean']), 'ssim_mean': float(row['ssim_mean'])}
    return grid


def write_table(
    grid: Dict[Tuple[int, int], Dict[str, float]], input_mode: str,
    metric_key: str, metric_label: str, fmt: str, output_path: pathlib.Path,
    ) -> None:
    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\begin{tabular}{l' + 'c' * len(N_MODULES) + '}')
    lines.append('\\toprule')
    lines.append('Connectivity & ' + ' & '.join(f'$n_{{\\text{{modules}}}}={n}$' for n in N_MODULES) + ' \\\\')
    lines.append('\\midrule')
    for connectivity in CONNECTIVITIES:
        cells = []
        for n_modules in N_MODULES:
            value = grid.get((connectivity, n_modules), {}).get(metric_key)
            cells.append(f'{value:{fmt}}' if value is not None else '--')
        lines.append(f'{connectivity} & ' + ' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append(
        f'\\caption{{{metric_label} vs.\\ connectivity and GLM module depth, {input_mode}, '
        'native resolution (no angular downsampling).}'
        )
    lines.append(f'\\label{{tab:depth-study-{input_mode}-{metric_key.split("_")[0]}}}')
    lines.append('\\end{table}')
    output_path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='directory for LaTeX tables')
    parser.add_argument(
        '--results-dir', default=DEFAULT_RESULTS_DIR,
        help='directory holding sinogram_denoising_depth_<input_mode>.csv, written by '
             'aggregate_sinogram_denoising_results.py',
        )
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    csv_paths = sorted(results_dir.glob('sinogram_denoising_depth_*.csv'))
    if not csv_paths:
        print(
            f'No sinogram_denoising_depth_*.csv found in {results_dir} -- run '
            'aggregate_sinogram_denoising_results.py first.'
            )
        return

    master_lines = []
    for csv_path in csv_paths:
        input_mode = csv_path.stem.removeprefix('sinogram_denoising_depth_')
        grid = read_depth_grid(csv_path)
        if not grid:
            print(f'No downsampling={NATIVE_DOWNSAMPLING} rows in {csv_path}, skipping {input_mode}')
            continue

        for metric_key, metric_stem, metric_label, fmt in METRICS:
            table_path = output_dir / f'depth_study_{input_mode}_{metric_stem}_table.tex'
            write_table(grid, input_mode, metric_key, metric_label, fmt, table_path)
            master_lines.append(f'\\input{{{table_path.name}}}')
            master_lines.append('')
        print(f'Wrote depth-study tables for {input_mode} ({len(grid)}/{len(CONNECTIVITIES) * len(N_MODULES)} configs found)')

    master_path = output_dir / 'depth_study.tex'
    master_path.write_text('\n'.join(master_lines))
    print(f'Wrote {master_path} (includes every depth-study table)')


if __name__ == '__main__':
    main()
