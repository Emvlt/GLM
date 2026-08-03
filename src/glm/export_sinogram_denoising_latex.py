"""
Exports sinogram-denoising `mode` family results (see
aggregate_sinogram_denoising_results.py) as paper-ready LaTeX figures and
tables: PSNR/SSIM vs. downsampling. One combined figure covers every
(input_mode, target_mode) pair as a pgfplots groupplot (rows = mode pairs,
columns = PSNR/SSIM), sharing the x-axis and a single legend across the
whole grid -- configs are identical across mode pairs, so there's exactly
one legend for the whole figure rather than one per pair. Tables stay
per-pair, one PSNR + one SSIM table per (input_mode, target_mode).

Uses median + per-seed scatter rather than mean +/- std. Several configs
have bimodal seed outcomes (e.g. GLM/16ch, mode2->mode2: seeds land at
either ~14 or ~42 PSNR, nothing in between) where a mean +/- std band is
actively misleading -- it reports ~31 +/- 15, which looks like uniform
uncertainty but is actually "works" or "diverges", never in between. Median
tracks the majority cluster and the per-seed scatter makes the split
visible instead of averaging it away. Tables report median [min, max] and
flag cells where the range is wide relative to the median, since that is
the signature of this failure mode rather than ordinary seed noise.

By default diverged seeds are kept and marked (red x on the chart, dagger
in the tables). Pass --exclude-diverged to drop them from both entirely --
useful once you're going to describe the divergence in prose instead of
showing it; the dropped seeds are printed so you know what to write about.

Reads the raw per-seed records that aggregate_sinogram_denoising_results.py
writes to `results/sinogram_denoising_records.json` -- run that script first
(it drives the dvc/git experiment walk) so this one can stay a plain,
fast read of results/ with no dvc dependency.

Figures are plain hand-shaped pgfplots source (`\\begin{axis}`/`\\nextgroupplot`
/ `\\addplot coordinates {...}`), not a matplotlib export: colors pulled out
into a handful of `\\definecolor` lines at the top, nothing else styled or
positioned beyond what `group style` needs for the shared axis/legend. The
point is to keep it editable -- restyle markers, colors, legend placement,
etc. directly in the .tex during final passes rather than regenerating from
Python. The combined figure is also printed to stdout as it's generated, to
eyeball/copy without opening a file. The paper's preamble needs
`\\usepackage{pgfplots}`, `\\usepgfplotslibrary{groupplots}`, and
`\\pgfplotsset{compat=1.18}`.
"""
import argparse
import json
import pathlib
import random
from collections import defaultdict
from statistics import median
from typing import Dict, List, Tuple

from glm.aggregate_sinogram_denoising_results import DEFAULT_DOWNSAMPLING_SWEEP, RECORDS_FILENAME

DEFAULT_OUTPUT_DIR = 'results/latex'
DEFAULT_RECORDS_PATH = f'results/{RECORDS_FILENAME}'
METRICS = (('psnr', 'PSNR (dB)', '.2f'), ('ssim', 'SSIM', '.3f'))
HIGH_SPREAD_RELATIVE_RANGE = 0.15  # (max-min)/|median| above this gets flagged

def latex_escape(text: str) -> str:
    return text.replace('_', '\\_')

def config_label_tex(active_model: str, n_channels) -> str:
    model = 'CNN' if active_model == 'sinogram_CNN' else latex_escape(active_model)
    return f'{model} ({n_channels})'

def diverged_seeds(records: List[Dict], metric_key: str = 'psnr') -> set:
    """Seeds whose per-downsampling median deviates from the config's
    per-seed median by more than HIGH_SPREAD_RELATIVE_RANGE.

    Catches the bimodal-outcome case described above: a seed whose model
    landed in the diverged/failed cluster rather than the majority one, as
    opposed to ordinary run-to-run noise. `records` should span every
    downsampling for a single (active_model, n_channels) config.
    """
    by_seed = defaultdict(list)
    for r in records:
        by_seed[r['seed']].append(r[metric_key])
    if len(by_seed) < 2:
        return set()
    seed_medians = {seed: median(vals) for seed, vals in by_seed.items()}
    group_median = median(seed_medians.values())
    if group_median == 0:
        return set()
    return {
        seed for seed, value in seed_medians.items()
        if abs(value - group_median) / abs(group_median) > HIGH_SPREAD_RELATIVE_RANGE
        }

def filter_diverged_records(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Splits records (for a single mode pair, any mix of configs) into
    (kept, excluded) by re-running diverged_seeds() per config and dropping
    every record whose seed comes back flagged for its own config.

    Downstream code (render_combined_pgfplots, write_table) also calls
    diverged_seeds() on whatever it's given -- once the outliers are gone
    from `kept`, that recheck naturally finds nothing left to flag, so no
    red-×/dagger marking shows up for records that were excluded here.
    """
    grouped = group_by_config_and_downsampling(records)
    excluded_seeds: Dict[Tuple[str, int], set] = {}
    for config, by_downsampling in grouped.items():
        config_records = [r for recs in by_downsampling.values() for r in recs]
        seeds = diverged_seeds(config_records)
        if seeds:
            excluded_seeds[config] = seeds

    kept, excluded = [], []
    for r in records:
        config = (r['active_model'], r['n_channels'])
        (excluded if r['seed'] in excluded_seeds.get(config, set()) else kept).append(r)
    return kept, excluded

def collect_mode_records(records_path: pathlib.Path) -> Dict[Tuple[str, str], List[Dict]]:
    """{(input_mode, target_mode): [record, ...]}, records keep every seed."""
    if not records_path.exists():
        raise FileNotFoundError(
            f'{records_path} not found -- run aggregate_sinogram_denoising_results.py first '
            'to generate it.'
            )
    records = json.loads(records_path.read_text())
    by_mode_pair = defaultdict(list)
    for record in records:
        if record['family'] != 'mode':
            continue
        by_mode_pair[(record['input_mode'], record['target_mode'])].append(record)
    return by_mode_pair

def group_by_config_and_downsampling(
    records: List[Dict],
    ) -> Dict[Tuple[str, int], Dict[int, List[Dict]]]:
    """{(active_model, n_channels): {downsampling: [record, ...]}}"""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[(r['active_model'], r['n_channels'])][r['downsampling']].append(r)
    return grouped

# Plain RGB triples, no matplotlib dependency -- edit/extend directly if a
# mode pair ever has more than len(PALETTE) configs.
PALETTE: List[Tuple[int, int, int]] = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    ]

def color_names_for_configs(configs: List[Tuple[str, int]]) -> Dict[Tuple[str, int], str]:
    return {config: f'seriesColor{i}' for i, config in enumerate(configs)}

def render_axis_body(
    grouped: Dict[Tuple[str, int], Dict[int, List[Dict]]],
    configs: List[Tuple[str, int]],
    colors: Dict[Tuple[str, int], str],
    diverged: Dict[Tuple[str, int], set],
    downsamplings: List[int],
    metric_key: str,
    rng: random.Random,
    *,
    add_legend: bool,
    ) -> List[str]:
    """The \\addplot/\\addlegendentry lines for one subplot -- no
    \\begin{axis}/\\nextgroupplot wrapper, so callers can use either."""
    lines = []
    for config in configs:
        color = colors[config]
        by_downsampling = grouped[config]
        diverged_seed_set = diverged.get(config, set())

        normal_coords = []
        diverged_coords = []
        for d in downsamplings:
            for r in by_downsampling.get(d, []):
                x = d + rng.uniform(-0.15, 0.15)
                coord = f'({x:.3f},{r[metric_key]:.4f})'
                (diverged_coords if r['seed'] in diverged_seed_set else normal_coords).append(coord)
        if normal_coords:
            lines.append(
                f'\\addplot[only marks, mark size=1pt, color={color}, opacity=0.4, '
                f'forget plot] coordinates {{{" ".join(normal_coords)}}};'
                )
        if diverged_coords:
            lines.append(
                '\\addplot[only marks, mark=x, mark size=2pt, color=red, '
                f'forget plot] coordinates {{{" ".join(diverged_coords)}}};'
                )

        median_coords = [
            f'({d},{median(r[metric_key] for r in by_downsampling[d]):.4f})'
            for d in downsamplings if by_downsampling.get(d)
            ]
        if median_coords:
            lines.append(f'\\addplot[mark=*, color={color}] coordinates {{{" ".join(median_coords)}}};')
            if add_legend:
                lines.append(f'\\addlegendentry{{{config_label_tex(*config)}}}')
    return lines

def render_combined_pgfplots(by_mode_pair: Dict[Tuple[str, str], List[Dict]]) -> str:
    """One pgfplots groupplot spanning every mode pair: rows are
    (input_mode, target_mode) pairs, columns are PSNR/SSIM. The x-axis is
    shared (ticks/label only on the bottom row) and there's a single legend
    for the whole thing -- configs/colors are identical across mode pairs,
    so one legend (defined on the very first subplot via `legend to name`,
    referenced once via `\\ref` below the grid) covers every row.

    Requires \\usepgfplotslibrary{groupplots} in the preamble, in addition to
    \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}. width/height/sep
    below are sized for a single-column page -- tighten them for a
    two-column layout.
    """
    mode_pairs = sorted(by_mode_pair.keys())
    rows = []
    for input_mode, target_mode in mode_pairs:
        grouped = group_by_config_and_downsampling(by_mode_pair[(input_mode, target_mode)])
        configs = sorted(grouped.keys())
        diverged = {
            config: diverged_seeds([r for recs in grouped[config].values() for r in recs])
            for config in configs
            }
        rows.append((input_mode, target_mode, grouped, configs, diverged))
    any_diverged = any(seeds for *_, diverged in rows for seeds in diverged.values())

    all_configs = sorted({config for *_, configs, _ in rows for config in configs})
    colors = color_names_for_configs(all_configs)
    legend_name = 'legendAll'

    lines = [
        f'\\definecolor{{{colors[config]}}}{{RGB}}{{{",".join(str(c) for c in PALETTE[i % len(PALETTE)])}}}'
        for i, config in enumerate(all_configs)
        ]
    lines.append('')
    lines.append('\\begin{tikzpicture}')
    lines.append('\\begin{groupplot}[')
    lines.append('    group style={')
    lines.append(f'        group size=2 by {len(rows)},')
    lines.append('        horizontal sep=1.3cm,')
    lines.append('        vertical sep=0.8cm,')
    lines.append('        xlabels at=edge bottom,')
    lines.append('        xticklabels at=edge bottom,')
    lines.append('    },')
    lines.append('    width=0.47\\linewidth,')
    lines.append('    height=4cm,')
    lines.append(f'    xtick={{{",".join(str(d) for d in DEFAULT_DOWNSAMPLING_SWEEP)}}},')
    lines.append('    xlabel={Downsampling factor},')
    lines.append(']')

    rng = random.Random(0)
    for row_i, (input_mode, target_mode, grouped, configs, diverged) in enumerate(rows):
        for col, (metric_key, metric_label, _) in enumerate(METRICS):
            is_first_axis = row_i == 0 and col == 0
            opts = [f'    ylabel={{{metric_label}}},']
            if col == 0:
                opts.append(f'    title={{{input_mode} $\\rightarrow$ {target_mode}}},')
            if is_first_axis:
                opts.append(f'    legend to name={legend_name},')
                opts.append('    legend columns=-1,')
            lines.append('\\nextgroupplot[')
            lines += opts
            lines.append(']')
            lines += render_axis_body(
                grouped, configs, colors, diverged, DEFAULT_DOWNSAMPLING_SWEEP, metric_key, rng,
                add_legend=is_first_axis,
                )
            if is_first_axis and any_diverged:
                lines.append('\\addlegendimage{only marks, mark=x, mark size=2pt, color=red}')
                lines.append('\\addlegendentry{Diverged seed}')

    lines.append('\\end{groupplot}')
    lines.append('\\end{tikzpicture}')
    lines.append('')
    lines.append('\\begin{center}')
    lines.append(f'\\ref{{{legend_name}}}')
    lines.append('\\end{center}')

    return '\n'.join(lines) + '\n'

def format_cell(values: List[float], fmt: str) -> str:
    if not values:
        return '--'
    lo, hi, med = min(values), max(values), median(values)
    spread_flag = ''
    if med != 0 and (hi - lo) / abs(med) > HIGH_SPREAD_RELATIVE_RANGE:
        spread_flag = '$^\\dagger$'
    cell = f'{med:{fmt}}{spread_flag}'
    if len(values) > 1:
        cell += f' \\scriptsize{{[{lo:{fmt}}, {hi:{fmt}}]}}'
    return cell

def write_table(
    input_mode: str, target_mode: str, metric_key: str, metric_label: str, fmt: str,
    records: List[Dict], output_path: pathlib.Path,
    ) -> None:
    grouped = group_by_config_and_downsampling(records)
    configs = sorted(grouped.keys())
    downsamplings = DEFAULT_DOWNSAMPLING_SWEEP
    max_n_seeds = max(
        (len(grouped[c][d]) for c in configs for d in downsamplings if grouped[c][d]),
        default=0,
        )

    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\resizebox{\\linewidth}{!}{%')
    lines.append('\\begin{tabular}{l' + 'c' * len(downsamplings) + '}')
    lines.append('\\toprule')
    lines.append('Model & ' + ' & '.join(f'$\\times${d}' for d in downsamplings) + ' \\\\')
    lines.append('\\midrule')
    for config in configs:
        row_n_seeds = {len(grouped[config][d]) for d in downsamplings if grouped[config][d]}
        label = config_label_tex(*config)
        if row_n_seeds and row_n_seeds != {max_n_seeds}:
            label += f' ($n$={min(row_n_seeds)})'
        cells = [
            format_cell([r[metric_key] for r in grouped[config][d]], fmt)
            for d in downsamplings
            ]
        lines.append(label + ' & ' + ' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('}')
    lines.append(
        f'\\caption{{{metric_label} (median [min, max] across seeds) vs.\\ downsampling '
        f'factor, {input_mode} $\\rightarrow$ {target_mode}. '
        '$^\\dagger$ range exceeds '
        f'{int(HIGH_SPREAD_RELATIVE_RANGE * 100)}\\% of the median -- seeds did not '
        'converge consistently; see the corresponding figure.}'
        )
    lines.append(f'\\label{{tab:sinogram-denoising-{input_mode}-{target_mode}-{metric_key}}}')
    lines.append('\\end{table}')
    output_path.write_text('\n'.join(lines) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='directory for LaTeX figures/tables')
    parser.add_argument(
        '--records-path', default=DEFAULT_RECORDS_PATH,
        help='raw per-seed records JSON written by aggregate_sinogram_denoising_results.py',
        )
    parser.add_argument(
        '--exclude-diverged', action='store_true',
        help=(
            'drop diverged seeds (see diverged_seeds()) from the chart and tables '
            'entirely, instead of marking them with a red x. Prints exactly which '
            'seeds got dropped, for writing up separately.'
            ),
        )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    by_mode_pair = collect_mode_records(pathlib.Path(args.records_path))
    if not by_mode_pair:
        print('No mode-family experiments with readable metrics found.')
        return

    if args.exclude_diverged:
        filtered = {}
        for (input_mode, target_mode), records in sorted(by_mode_pair.items()):
            kept, excluded = filter_diverged_records(records)
            filtered[(input_mode, target_mode)] = kept
            for active_model, n_channels in sorted({(r['active_model'], r['n_channels']) for r in excluded}):
                seeds = sorted({r['seed'] for r in excluded
                                if r['active_model'] == active_model and r['n_channels'] == n_channels})
                print(
                    f'Excluding {len(seeds)} diverged seed(s) for {input_mode} -> {target_mode}, '
                    f'{config_label_tex(active_model, n_channels)}: {seeds}'
                    )
        by_mode_pair = filtered

    fig_path = output_dir / 'sinogram_denoising_combined.tex'
    fig_tex = render_combined_pgfplots(by_mode_pair)
    fig_path.write_text(fig_tex)
    print('\n%% --- combined figure (all mode pairs) ---')
    print(fig_tex)

    master_lines = []
    master_lines.append('\\begin{figure}[t]')
    master_lines.append('\\centering')
    master_lines.append(f'\\input{{{fig_path.name}}}')
    if args.exclude_diverged:
        master_lines.append(
            '\\caption{PSNR and SSIM vs.\\ downsampling factor, for every input/target mode '
            'pair. Dots are individual seeds (jittered horizontally for visibility); lines '
            'connect the per-downsampling median. Seeds that diverged by more than '
            f'{int(HIGH_SPREAD_RELATIVE_RANGE * 100)}\\% from the config median (likely '
            'training divergence rather than ordinary noise) are excluded here -- see '
            'discussion.}'
            )
    else:
        master_lines.append(
            '\\caption{PSNR and SSIM vs.\\ downsampling factor, for every input/target mode '
            'pair. Dots are individual seeds (jittered horizontally for visibility); lines '
            'connect the per-downsampling median. Seeds whose result diverges by more than '
            f'{int(HIGH_SPREAD_RELATIVE_RANGE * 100)}\\% from the config median (likely '
            'training divergence rather than ordinary noise) are marked with a red '
            '$\\times$ instead of a dot.}'
            )
    master_lines.append('\\label{fig:sinogram-denoising-all}')
    master_lines.append('\\end{figure}')
    master_lines.append('')
    print(f'Wrote {fig_path.name}')

    for (input_mode, target_mode), records in sorted(by_mode_pair.items()):
        stem = f'sinogram_denoising_{input_mode}_{target_mode}'
        for metric_key, metric_label, fmt in METRICS:
            table_path = output_dir / f'{stem}_{metric_key}_table.tex'
            write_table(input_mode, target_mode, metric_key, metric_label, fmt, records, table_path)
            master_lines.append(f'\\input{{{table_path.name}}}')
            master_lines.append('')
        print(f'Wrote tables for {input_mode} -> {target_mode}')

    (output_dir / 'results.tex').write_text('\n'.join(master_lines))
    print(f'Wrote {output_dir / "results.tex"} (includes the combined figure and all tables)')

if __name__ == '__main__':
    main()
