"""Verify and render C8 stage endpoints against one fixed Omni reference."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/data/c8-optimization-journey.json'
STEM = ROOT / 'docs/figures/npu-c8-optimization-journey'
METRICS = ('ttfp_ms', 'rtf', 'request_s')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def source_values(source):
    path = ROOT / source['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], path
    value = read(path)
    for key in source['selector']:
        value = value[key]
    assert value['completed'] == 128 and value['failed'] == 0
    assert value[source['input_tokens_field']] == 17646
    assert value[source['concurrency_field']] == 8
    return {name: value[field] for name, field in source['metric_fields'].items()}


def verify(data):
    assert data['concurrency'] == 8 and data['requests_per_source_run'] == 128
    assert data['speech_gate'] == 'NOT_RUN' and not data['equal_quality_claim']
    assert [stage['tag'] for stage in data['stages']] == [
        'r187', 'r215', 'r239', 'r247', 'r258', 'r292', 'r305', 'r309', 'r337']
    for item in [data['baseline'], *data['stages']]:
        values = [source_values(source) for source in item['sources']]
        for key in METRICS:
            expected = sum(value[key] for value in values) / len(values)
            assert math.isfinite(expected) and expected > 0
            assert math.isclose(expected, item['metrics'][key], abs_tol=1e-12)
    current = read(ROOT / 'docs/data/ascend-r337-20261103.json')['rows'][-1]
    for key, native in (('ttfp_ms', 'mean_audio_ttfp_ms'), ('rtf', 'mean_audio_rtf'),
                        ('request_s', 'request_throughput')):
        assert data['baseline']['metrics'][key] == current['omni'][native]
        assert data['stages'][-1]['metrics'][key] == current['engine_result'][native]
    r292 = next(stage for stage in data['stages'] if stage['tag'] == 'r292')
    previous = data['stages'][4]
    assert all(stage_gain(previous['metrics'][key], r292['metrics'][key], key) < 0 for key in METRICS)
    return sum(len(item['sources']) for item in [data['baseline'], *data['stages']])


def stage_gain(before, after, key):
    return 100 * (after / before - 1) if key == 'request_s' else 100 * (1 - after / before)


def ratio(baseline, value, key):
    return value / baseline if key == 'request_s' else baseline / value


def write_document(data):
    baseline = data['baseline']['metrics']
    rows = []
    for stage in data['stages']:
        metrics = stage['metrics']
        multiples = [ratio(baseline[key], metrics[key], key) for key in METRICS]
        rows.append('| ' + ' | '.join((stage['tag'], stage['optimization'], f"{metrics['ttfp_ms']:.2f}",
            f"{metrics['rtf']:.4f}", f"{metrics['request_s']:.4f}",
            *[f'{value:.3f}×' for value in multiples])) + ' |')
    changes = []
    for previous, stage in zip(data['stages'], data['stages'][1:]):
        gains = [stage_gain(previous['metrics'][key], stage['metrics'][key], key) for key in METRICS]
        changes.append('| ' + ' | '.join((previous['tag'] + ' → ' + stage['tag'],
            *[f'{value:+.2f}%' for value in gains])) + ' |')
    first = data['stages'][0]['metrics']
    last = data['stages'][-1]['metrics']
    total = [stage_gain(first[key], last[key], key) for key in METRICS]
    stage_rows = '\n'.join(rows)
    change_rows = '\n'.join(changes)
    text = f'''# 以C8看完整优化历程

**历史测量。当前代码已移除 prompt-lookup，移除后的性能待重测。下图 r305/r309/r337 点对应含 lookup 的归档版本。** [移除记录](../evidence/20261103-lookup-removal/README.md)

## 从起点到当前

横轴按九个关键版本累计展示优化内容。当前Omni main9990f10d9的C8固定为1.00×，三项性能倍数均为越高越好。

![C8完整优化历程](figures/npu-c8-optimization-journey.svg)

本图对照固定为TTFP {baseline['ttfp_ms']:.2f}ms、RTF {baseline['rtf']:.4f}、吞吐{baseline['request_s']:.4f}request/s。TTFP速度倍数按Omni TTFP÷引擎TTFP计算，RTF效率倍数按Omni RTF÷引擎RTF计算，吞吐倍数按引擎request/s÷Omni request/s计算。

从r187到r337，阶段端点的TTFP降低{total[0]:.2f}%，RTF降低{total[1]:.2f}%，吞吐提高{total[2]:.2f}%。当前r337使用与四档产品对比图相同的两次运行均值。

## 每个点做了什么

| 版本 | 累计阶段的优化内容 | TTFP mean ms | RTF mean | request/s | TTFP速度倍数 | RTF效率倍数 | 吞吐倍数 |
|---|---|---|---|---|---|---|---|
{stage_rows}

## 每段拿到多少变化

表中百分比按相邻阶段的原生指标计算。正值对应TTFP/RTF降低或吞吐增加。跨日期端点包含组合改动和运行波动，单项机制的贡献另看当期匹配对照。

| 相邻阶段 | TTFP降低 | RTF降低 | request/s增加 |
|---|---|---|---|
{change_rows}

## 如何读这条历史曲线

- r32–r181完成NPU迁移、采样/图/流式状态和旧冻结协议建设。原生官方CLI可直接串联的起点是r187，旧TTFA-p50/frames/s证据保留在历史账本。
- r187–r239阶段包含K10完整设备块、状态所有权修正、Add+RMS与CFM hybrid筛选。r224–r226的质量失败、packed-QKV正式服务可靠性失败均保留在完整记录。
- r247补选择性batch图。r258增加prefix KV与压力门控，主要面向高压力。该版本在C8的点属于复测端点，其差值尚未独立归因到prefix命中。
- r292的C8三指标相对匹配r258均回退，图中保留这段下降。公平capture gate的等待会在不同执行者之间迁移。
- r305通过Thinker prompt-lookup及原模型块验证减少首文本串行等待。主图相邻历史端点的收益与其独立当期控制收益分别记录，完整控制表见六层分析报告。
- r309在真实audio worker和prompt stream预热Euler/HiFT桶。该点取最后匹配确认窗口的三次control309均值。它在历史端点对比中出现RTF/吞吐回退。
- r337结合本进程12Cube/24Vector与可信累计query长度复用。最后40轮还测试了私有Flow、更多图桶、C++派发、RMS/RoPE等方案。候选在40轮上限处保留，其余失败和无一致收益结果也有逐项记录。

各点均来自单Ascend910B1/A2上的C8、128个计时请求、0请求失败、17646输入token。早期阶段取一次原生观测，r309取三次控制均值，r337取两次候选均值。主图统一使用当前固定Omni参考，历史报告中的当期Omni基线也各自保留。

Omni Flow为FP32/10步，引擎为FP16/6步，RNG及生成音频时长存在差异。完整当前语音质量门仍待完成。

## 数据、复算和完整记录

```bash
python tools/plot_c8_optimization_journey.py --verify-only
python tools/plot_c8_optimization_journey.py
```

- [PNG](figures/npu-c8-optimization-journey.png) · [SVG](figures/npu-c8-optimization-journey.svg) · [PDF](figures/npu-c8-optimization-journey.pdf)
- [源数据、逐点SHA与聚合口径](data/c8-optimization-journey.json)
- [早期RSI到6层profiling 诊断的历程](optimization-journey.zh-CN.md)
- [全NPU账本](optimization-ledger-npu.zh-CN.md)
- [r337最新总结](rsi-r337-20261103.zh-CN.md)
- [六层分析与r305匹配控制](rsi-profiling-20261102.zh-CN.md)
- [后续40轮逐项记录](../evidence/20261103-r337/campaign/持续优化实验记录.md)
'''
    (ROOT / 'docs/c8-optimization-journey.zh-CN.md').write_text(text.strip() + '\n', encoding='utf-8', newline='\n')


def render(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import FuncFormatter

    font = FontProperties(family=['Microsoft YaHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei'])
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                         'font.size': 12, 'svg.fonttype': 'path', 'pdf.fonttype': 42})
    stages = data['stages']
    baseline = data['baseline']['metrics']
    first = stages[0]['metrics']
    last = stages[-1]['metrics']
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(19.5, 12.8))
    fig.subplots_adjust(left=.065, right=.975, top=.81, bottom=.235, hspace=.49)
    fig.suptitle('以 C8 看独立引擎的完整优化历程', y=.986, fontsize=25, color='#183545', weight='bold')
    fig.text(.5, .935, '固定本次 vLLM-Omni main@9990f10d9 的 C8 = 1.00×，纵轴均为越高越好',
             ha='center', fontsize=14, color='#314e5f')
    fig.text(.5, .905, '单 Ascend 910B1 / A2 ｜ 官方 CLI · 每个源运行 128 请求 ｜ 边注显示相邻阶段收益',
             ha='center', fontsize=12, color='#5c6d78')

    phases = [(-.4, .5, '初版', '#e9edf0'), (.5, 2.5, 'RSI 机制适配', '#e8f2e9'),
              (2.5, 5.5, 'profiling ＋ 自主策略', '#e7eff7'), (5.5, 6.5, '6层profiling 诊断', '#eee8f7'),
              (6.5, 8.4, '持续 profiling / 后续 40 轮', '#fbede2')]
    for left, right, name, color in phases:
        position = .065 + .91 * (((left + right) / 2 + .4) / 8.8)
        fig.text(position, .865, name, ha='center', fontsize=11.5, color='#405561',
                 bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=.4'))
    settings = [
        ('ttfp_ms', '首包速度倍数  ·  Omni TTFP / 引擎 TTFP', '#126b9a', 'ms', 1),
        ('rtf', 'RTF 效率倍数  ·  Omni RTF / 引擎 RTF', '#7953a5', '', 4),
        ('request_s', '请求吞吐倍数  ·  引擎 request/s / Omni request/s', '#c66b2d', 'request/s', 4),
    ]
    for ax, (key, title, color, unit, precision) in zip(axes, settings):
        values = [ratio(baseline[key], stage['metrics'][key], key) for stage in stages]
        for left, right, _, fill in phases:
            ax.axvspan(left, right, color=fill, alpha=.45, zorder=0)
        ax.axhline(1, color='#798b95', lw=1.2, ls='--', zorder=1)
        ax.text(8.32, 1.015, 'Omni 1.00×', color='#677d89', ha='right', fontsize=10)
        ax.plot(range(len(stages)), values, 'o-', color=color, lw=2.7, ms=7, zorder=3)
        for index, value in enumerate(values):
            ax.annotate(f'{value:.3f}×', (index, value), xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=12, color=color, weight='bold')
        for index in range(1, len(stages)):
            gain = stage_gain(stages[index - 1]['metrics'][key], stages[index]['metrics'][key], key)
            y = max(.60, (values[index - 1] + values[index]) / 2 - .15)
            ax.text(index - .5, y, f'{gain:+.1f}%', ha='center', va='center', fontsize=11,
                    color='#2a7c5e' if gain >= 0 else '#b34b43',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1.5))
        ax.set_title(title, loc='left', pad=18, fontsize=14, color=color, weight='bold')
        overall = stage_gain(first[key], last[key], key)
        direction = '提高' if key == 'request_s' else '降低'
        ax.text(1, 1.12, f"{first[key]:.{precision}f} → {last[key]:.{precision}f} {unit}   累计{direction} {overall:.1f}%",
                transform=ax.transAxes, ha='right', fontsize=12, color='#425c6a')
        ax.set_xlim(-.4, 8.4)
        ax.set_ylim(.55, 1.5)
        ax.set_yticks((.75, 1, 1.25, 1.5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.2f}×'))
        ax.grid(axis='y', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#afbdc5')
        ax.tick_params(colors='#4a606d')
    axes[-1].set_xticks(range(len(stages)), [stage['label'] for stage in stages], fontsize=11.5)
    axes[-1].tick_params(axis='x', pad=12)
    axes[-1].set_xlabel('累计演进的关键优化阶段', labelpad=15, fontsize=13)
    footnote = (
        '边注为相邻端点的 TTFP / RTF 降低或 request/s 提高。跨日期端点包含组合改动和运行波动。\n'
        'r309 取最终窗口三次控制均值，r337 取两次候选均值。r292 的 C8 回退与 r309 的混合结果如实保留。\n'
        '前置 r32–r181 完成 NPU 迁移与旧协议建设，详见完整账本。当前图从统一官方 CLI 的 r187 开始。\n'
        'Flow 引擎 FP16 / 6步，Omni FP32 / 10步，RNG 与生成音频时长不同。完整语音质量门待完成。')
    fig.text(.065, .025, footnote, ha='left', va='bottom', fontsize=10.5, color='#5a6c77', linespacing=1.55)
    for suffix in ('png', 'svg', 'pdf'):
        metadata = {'Date': None} if suffix == 'svg' else (
            {'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else {})
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
    sources = verify(data)
    if not args.verify_only:
        render(data)
        write_document(data)
    print(json.dumps(dict(stages=len(data['stages']), source_cells=sources, concurrency=8,
                          fixed_baseline=data['baseline']['name'], source_hashes_verified=True,
                          regression_preserved=True, speech_gate='NOT_RUN')))


if __name__ == '__main__':
    main()
