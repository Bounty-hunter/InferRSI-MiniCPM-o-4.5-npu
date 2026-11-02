"""Render the recorded Ascend campaign; never treat plots as new measurements."""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    ('ttfp_ms', 'Time to first audio', 'TTFP (ms) · lower is better'),
    ('rtf', 'Real-time factor', 'RTF · lower is better'),
    ('req_s', 'Request throughput', 'request/s · higher is better'),
)
SERIES = (
    ('historical_omni', 'vLLM-Omni (vLLM 0.32.0)', '#2563eb', '--', '^'),
    ('candidate', 'Dedicated NPU engine (r292 candidate)', '#ea580c', '-', 'o'),
)


def configure_axes(ax, x):
    ax.set_xticks(x, [f'C{v}' for v in x])
    ax.set_xlabel('Concurrency', labelpad=9)
    ax.grid(axis='y', color='#dbe3ec', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#cbd5e1')
    ax.tick_params(axis='both', length=0, pad=7)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.margins(x=0.07)


def finish(fig, axes, title, subtitle, legend, footnote, output):
    fig.suptitle(title, x=0.045, y=0.985, ha='left', fontsize=21, fontweight='bold', color='#0f172a')
    fig.text(0.045, 0.909, subtitle, fontsize=10.5, color='#475569')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.04, 0.88),
               ncol=legend, frameon=False, fontsize=10)
    fig.text(0.045, 0.055, footnote, fontsize=9, color='#475569', linespacing=1.65)
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.24, top=0.72, wspace=0.30)
    for suffix in ('svg', 'png'):
        path = output.with_suffix('.'+suffix)
        metadata = {'Creator': 'tools/plot_npu_performance.py'}
        if suffix == 'svg':
            metadata['Date'] = None
        fig.savefig(path, dpi=190, facecolor='white', metadata=metadata)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',
                            encoding='utf-8', newline='\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'docs/data/ascend-r295-comparison.json')
    parser.add_argument('--out-dir', type=Path, default=ROOT/'docs/figures')
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding='utf-8'))
    rows = data['measurements']
    x = [row['concurrency'] for row in rows]
    assert x == [1, 4, 8, 16, 32]
    for row in rows:
        for variant in ('candidate', 'control', 'historical_omni'):
            assert row[variant]['failed'] == 0
        for reference, key in (('control', 'matched_r258'), ('historical_omni', 'historical_omni')):
            for metric, *_ in METRICS:
                a, b = row[reference][metric], row['candidate'][metric]
                gain = (b/a-1)*100 if metric == 'req_s' else (1-b/a)*100
                assert math.isclose(gain, row['benefit_percent'][key][metric], abs_tol=1e-9)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelcolor': '#334155', 'text.color': '#0f172a',
                         'xtick.color': '#475569', 'ytick.color': '#475569',
                         'svg.fonttype': 'none', 'svg.hashsalt': 'minicpmo-ascend-r295'})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (metric, title, ylabel) in zip(axes, METRICS):
        for variant, label, color, linestyle, marker in SERIES:
            ax.plot(x, [r[variant][metric] for r in rows], label=label, color=color,
                    linestyle=linestyle, marker=marker, linewidth=2.4,
                    markersize=6, markeredgecolor='white', markeredgewidth=0.8)
        configure_axes(ax, x)
        ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        if metric == 'ttfp_ms':
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    finish(fig, axes, 'MiniCPM-o 4.5 | Ascend performance curves',
           'One Ascend 910B1 (A2) per service · Seed-TTS English · chunk25 · 32 / 64 / 128 / 256 / 512 requests', 2,
           'Omni: main@5b5eb49c, vLLM 0.32.0; recorded Oct 26–28. Engine: r292 candidate, measured Oct 30. Both: 0 request errors.\n'
           'Previously frozen Omni baseline, not a simultaneous rerun. Single-run points; no confidence intervals / frozen LCB or final speech-gate pass.',
           args.out_dir/'npu-performance-curves')

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    titles = ('TTFP reduction', 'RTF reduction', 'Throughput increase')
    for ax, (metric, *_), title in zip(axes, METRICS, titles):
        for key, label, color, linestyle, marker in (
            ('historical_omni', 'Dedicated NPU engine vs. vLLM-Omni (vLLM 0.32.0)', '#ea580c', '-', 'o'),
        ):
            ax.plot(x, [r['benefit_percent'][key][metric] for r in rows],
                    label=label, color=color, linestyle=linestyle, marker=marker,
                    linewidth=2.4, markersize=6, markeredgecolor='white', markeredgewidth=0.8)
        configure_axes(ax, x)
        ax.axhline(0, color='#0f172a', linewidth=0.8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:+.0f}%'))
        ax.set_ylabel('Improvement (%) · positive is better')
        ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=12)
        for row in rows:
            value = row['benefit_percent']['historical_omni'][metric]
            ax.annotate(f'{value:+.1f}%', (row['concurrency'], value),
                        xytext=(0, 10 if value >= 0 else -16), textcoords='offset points',
                        ha='center', fontsize=9, color='#9a3412')
        ax.margins(y=0.22)
    finish(fig, axes, 'MiniCPM-o 4.5 | Relative improvements',
           'Independent NPU engine vs. vLLM-Omni · same Ascend workload · negative values show regressions', 1,
           'Latency / RTF: (1 − candidate/reference) × 100%; throughput: (candidate/reference − 1) × 100%.\n'
           'Omni: main@5b5eb49c with vLLM 0.32.0, recorded earlier. These are system comparisons, not isolated RSI gains or final acceptance.',
           args.out_dir/'npu-improvement-curves')
    print('Saved performance and improvement curves as SVG and PNG:', args.out_dir)


if __name__ == '__main__':
    main()
