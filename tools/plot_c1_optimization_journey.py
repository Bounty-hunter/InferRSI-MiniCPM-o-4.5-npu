"""Verify and render measured C1 endpoints, retaining unmeasured stage gaps."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from plot_c8_optimization_journey import METRICS, ratio, read, stage_gain

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/data/c1-optimization-journey.json'
STEM = ROOT / 'docs/figures/npu-c1-optimization-journey'


def verify(data):
    assert data['concurrency'] == 1 and data['requests_per_source_run'] == 32
    assert data['input_tokens_per_source_run'] == 4411
    assert data['speech_gate'] == 'NOT_RUN' and not data['equal_quality_claim']
    c8 = read(ROOT / 'docs/data/c8-optimization-journey.json')
    assert [row['tag'] for row in data['stages']] == [row['tag'] for row in c8['stages']]
    assert data['missing_stages'] == ['r247', 'r309']
    count = 0
    for item in [data['baseline'], *data['stages']]:
        if item.get('tag') in data['missing_stages']:
            assert item['metrics'] is None and item['sources'] == [] and item['aggregation'] == 'missing'
            continue
        values = []
        for source in item['sources']:
            path = ROOT / source['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], path
            raw = read(path)
            for key in source['selector']:
                raw = raw[key]
            assert raw['completed'] == 32 and raw['failed'] == 0
            assert raw[source['input_tokens_field']] == 4411
            assert raw[source['concurrency_field']] == 1
            values.append({key: raw[field] for key, field in source['metric_fields'].items()})
            count += 1
        for key in METRICS:
            expected = sum(value[key] for value in values) / len(values)
            assert math.isfinite(expected) and expected > 0
            assert math.isclose(expected, item['metrics'][key], rel_tol=1e-12)
    current = next(row for row in read(ROOT / 'docs/data/ascend-r337-20261103.json')['rows']
                   if row['concurrency'] == 1)
    for key, native in (('ttfp_ms', 'mean_audio_ttfp_ms'), ('rtf', 'mean_audio_rtf'),
                        ('request_s', 'request_throughput')):
        assert data['baseline']['metrics'][key] == current['omni'][native]
        assert data['stages'][-1]['metrics'][key] == current['engine_result'][native]
    folder = ROOT / 'evidence/optimization-journey/c1-milestones/r215'
    assert read(folder / 'exit-c1.json')['exit_code'] == 0
    assert read(folder / 'server-exit-code.json') == dict(returncode=0, cleanup_requested=True, forced=False)
    assert read(folder / 'info.json')['device_name'] == 'Ascend910B1'
    for name in ('info.json', 'config.json', 'status.json', 'input-sha256.json', 'runtime-import-audit.json'):
        assert (folder / name).read_bytes() == (folder.parents[1] / 'c8-milestones/r215' / name).read_bytes()
    return count


def write_document(data):
    baseline = data['baseline']['metrics']
    measured = [row for row in data['stages'] if row['metrics'] is not None]
    rows = []
    for stage in data['stages']:
        m = stage['metrics']
        values = ([f"{m['ttfp_ms']:.2f}", f"{m['rtf']:.4f}", f"{m['request_s']:.4f}",
                   *[f'{ratio(baseline[key], m[key], key):.3f}×' for key in METRICS]]
                  if m else ['缺测'] * 6)
        rows.append('| ' + ' | '.join([stage['tag'], stage['optimization'], *values]) + ' |')
    changes = []
    for previous, stage in zip(measured, measured[1:]):
        changes.append('| ' + ' | '.join([previous['tag'] + ' → ' + stage['tag'],
            *[f"{stage_gain(previous['metrics'][key], stage['metrics'][key], key):+.2f}%" for key in METRICS],
            '跨越缺测阶段' if (previous['tag'], stage['tag']) in (('r239', 'r258'), ('r305', 'r337')) else '相邻实测阶段']) + ' |')
    first, last = measured[0]['metrics'], measured[-1]['metrics']
    total = [stage_gain(first[key], last[key], key) for key in METRICS]
    text = f'''# 以 C1 看优化演进曲线

**历史测量。当前代码已移除 prompt-lookup，移除后的性能待重测。下图 r305/r337 点对应含 lookup 的归档版本。** [移除记录](../evidence/20261103-lookup-removal/README.md)

横轴沿用 C8 图的九个关键版本，包含七个有独立 C1 实测值的阶段。r247、r309 保留缺测位置，虚线连接前后实测端点。每个源运行均为 C1、32 个计时请求、4411 输入 token，观测请求错误为 0。

![C1优化演进曲线](figures/npu-c1-optimization-journey.svg)

## 固定对照与读图口径

固定 vLLM-Omni main@9990f10d9 的 C1 为 1.00×。对照值为 mean TTFP {baseline['ttfp_ms']:.2f} ms、mean RTF {baseline['rtf']:.4f}、吞吐 {baseline['request_s']:.4f} request/s。TTFP 速度倍数按 Omni TTFP ÷ 引擎 TTFP，RTF 效率倍数按 Omni RTF ÷ 引擎 RTF，吞吐倍数按引擎 request/s ÷ Omni request/s，纵轴均为越高越好。

从 r187 到 r337，历史端点的 mean TTFP 降低 {total[0]:.2f}%，mean RTF 降低 {total[1]:.2f}%，request/s 提高 {total[2]:.2f}%。r337 使用与当前四档产品对比图相同的两次 C1 运行均值，其余有值阶段各取一次原生观测。

## 各阶段实测值

| 版本 | 累计优化阶段 | TTFP mean ms | RTF mean | request/s | TTFP速度倍数 | RTF效率倍数 | 吞吐倍数 |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## 前后实测端点的变化

| 区间 | TTFP降低 | RTF降低 | request/s增加 | 记录范围 |
|---|---:|---:|---:|---|
{chr(10).join(changes)}

阶段变化包含组合改动和跨日期运行波动。r239 → r258 包含选择性 batch 图等区间改动，r305 → r337 包含媒体图预热、核配额和 host 长度复用等区间改动。缺测阶段的单项贡献仍无独立 C1 测量。r258 与 r292 使用 r295 中的匹配复测端点，C1 三项指标均有小幅改善，走势与 C8 不同。r305 有独立 C1 补测，缺少同轮 r292 C1 控制。

Omni Flow 为 FP32/10 步，引擎为 FP16/6 步，RNG 路径与生成音频时长存在差异。完整语音质量门仍为 `NOT_RUN`。核配额固定输入实验还观察到跨配额数值变化，详见[配额验证报告](core-quota-causality.zh-CN.md)。

## 数据与复核

```sh
python tools/plot_c1_optimization_journey.py --verify-only
python tools/plot_c1_optimization_journey.py
```

- [PNG](figures/npu-c1-optimization-journey.png) · [SVG](figures/npu-c1-optimization-journey.svg) · [PDF](figures/npu-c1-optimization-journey.pdf)
- [原始来源、SHA256与聚合口径](data/c1-optimization-journey.json)
- [C8演进曲线](c8-optimization-journey.zh-CN.md) · [完整优化历程](optimization-journey.zh-CN.md)
'''
    (ROOT / 'docs/c1-optimization-journey.zh-CN.md').write_text(text, encoding='utf-8', newline='\n')


def render(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import FuncFormatter

    font = FontProperties(family=['Microsoft YaHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei'])
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                         'font.size': 12, 'svg.fonttype': 'path', 'pdf.fonttype': 42})
    stages, baseline = data['stages'], data['baseline']['metrics']
    observed = [(index, stage) for index, stage in enumerate(stages) if stage['metrics'] is not None]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(19.5, 12.8))
    fig.subplots_adjust(left=.065, right=.975, top=.81, bottom=.235, hspace=.49)
    fig.suptitle('以 C1 看独立引擎的优化演进', y=.986, fontsize=25, color='#183545', weight='bold')
    fig.text(.5, .935, '固定 vLLM-Omni main@9990f10d9 的 C1 = 1.00×，纵轴均为越高越好',
             ha='center', fontsize=14, color='#314e5f')
    fig.text(.5, .905, '单 Ascend 910B1 / A2 ｜ 官方 CLI · 每个源运行 32 请求 ｜ 七个实测阶段 · 两处缺测',
             ha='center', fontsize=12, color='#5c6d78')
    phases = [(-.4, .5, '初版', '#e9edf0'), (.5, 2.5, 'RSI 机制适配', '#e8f2e9'),
              (2.5, 5.5, 'profiling ＋ 自主策略', '#e7eff7'), (5.5, 6.5, '6层profiling 诊断', '#eee8f7'),
              (6.5, 8.4, '持续 profiling / 后续 40 轮', '#fbede2')]
    for left, right, name, color in phases:
        position = .065 + .91 * (((left + right) / 2 + .4) / 8.8)
        fig.text(position, .865, name, ha='center', fontsize=11.5, color='#405561',
                 bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=.4'))
    settings = [('ttfp_ms', '首包速度倍数  ·  Omni TTFP / 引擎 TTFP', '#126b9a', 'ms', 2),
                ('rtf', 'RTF 效率倍数  ·  Omni RTF / 引擎 RTF', '#7953a5', '', 4),
                ('request_s', '请求吞吐倍数  ·  引擎 request/s / Omni request/s', '#c66b2d', 'request/s', 4)]
    for ax, (key, title, color, unit, precision) in zip(axes, settings):
        points = [(index, ratio(baseline[key], stage['metrics'][key], key)) for index, stage in observed]
        for left, right, _, fill in phases:
            ax.axvspan(left, right, color=fill, alpha=.45, zorder=0)
        ax.axhline(1, color='#798b95', lw=1.2, ls='--')
        ax.text(8.32, 1.015, 'Omni 1.00×', color='#677d89', ha='right', fontsize=10)
        for (before_index, before), (after_index, after) in zip(observed, observed[1:]):
            y0, y1 = [ratio(baseline[key], stage['metrics'][key], key) for stage in (before, after)]
            gap = after_index - before_index > 1
            ax.plot([before_index, after_index], [y0, y1], color=color, lw=2.7, ls='--' if gap else '-')
            gain = stage_gain(before['metrics'][key], after['metrics'][key], key)
            ax.text((before_index + after_index) / 2, (y0 + y1) / 2 - .12,
                    f'{gain:+.1f}%' + ('\n区间累计' if gap else ''), ha='center', va='center', fontsize=10.5,
                    color='#2a7c5e' if gain >= 0 else '#b34b43',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1.5))
        ax.plot([x for x, _ in points], [y for _, y in points], 'o', color=color, ms=7)
        for x, y in points:
            ax.annotate(f'{y:.3f}×', (x, y), xytext=(0, 10), textcoords='offset points',
                        ha='center', color=color, weight='bold', fontsize=12)
        first, last = stages[0]['metrics'][key], stages[-1]['metrics'][key]
        direction = '提高' if key == 'request_s' else '降低'
        ax.set_title(title, loc='left', pad=18, fontsize=14, color=color, weight='bold')
        ax.text(1, 1.12, f'{first:.{precision}f} → {last:.{precision}f} {unit}   累计{direction} {stage_gain(first, last, key):.1f}%',
                transform=ax.transAxes, ha='right', fontsize=12, color='#425c6a')
        ax.set_xlim(-.4, 8.4)
        ax.set_ylim(.70, max(y for _, y in points) + .20)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.2f}×'))
        ax.grid(axis='y', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#afbdc5')
        ax.tick_params(colors='#4a606d')
    labels = [stage['label'] + ('\n（C1缺测）' if stage['metrics'] is None else '') for stage in stages]
    axes[-1].set_xticks(range(len(stages)), labels, fontsize=11.5)
    axes[-1].tick_params(axis='x', pad=12)
    axes[-1].set_xlabel('累计演进的关键优化阶段', labelpad=15, fontsize=13)
    footnote = ('实线连接相邻实测阶段，虚线跨越缺测阶段。边注为前后实测端点变化，跨缺测标注为区间累计。\n'
                'r247、r309 未归档独立 C1 测量。r337 取两次 C1 均值，其余有值阶段各取一次原生观测。\n'
                '跨日期端点包含组合改动和运行波动。前置 r32–r181 迁移与旧协议建设详见完整账本。\n'
                'Flow 引擎 FP16 / 6步，Omni FP32 / 10步，RNG 与音频时长不同。完整语音质量门待完成。')
    fig.text(.065, .025, footnote, ha='left', va='bottom', fontsize=10.5, color='#5a6c77', linespacing=1.55)
    for suffix in ('png', 'svg', 'pdf'):
        metadata = {'Date': None} if suffix == 'svg' else ({'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else {})
        target = STEM.with_suffix('.' + suffix)
        fig.savefig(target, dpi=180, facecolor='white', metadata=metadata)
        if suffix == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text(encoding='utf-8').splitlines()) + '\n',
                              encoding='utf-8', newline='\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    data = read(DATA)
    count = verify(data)
    if not args.verify_only:
        render(data)
        write_document(data)
    print(json.dumps(dict(source_cells=count, measured_stages=7, missing_stages=data['missing_stages'],
                          concurrency=1, source_hashes_verified=True, speech_gate=data['speech_gate'])))


if __name__ == '__main__':
    main()
