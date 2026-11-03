"""Render illustrative diagrams for the r337 technical guide, without model loading."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures/adopted-optimizations'
BLUE = '#126b9a'
GREEN = '#267757'
ORANGE = '#bc672b'
GRAY = '#536875'


def canvas(title, subtitle, height=7.7):
    font = FontProperties(family=['Microsoft YaHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei'])
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                         'svg.fonttype': 'path', 'svg.hashsalt': 'adopted-optimizations-r337', 'font.size': 12})
    fig, ax = plt.subplots(figsize=(16, height))
    ax.set_position((.035, .12, .93, .73))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    fig.suptitle(title, y=.97, fontsize=23, weight='bold', color='#183545')
    fig.text(.5, .895, subtitle, ha='center', fontsize=12, color=GRAY)
    return fig, ax


def box(ax, x, y, width, height, title, body='', color=BLUE, fill='#edf5f9', size=13):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=.008,rounding_size=.014',
                              facecolor=fill, edgecolor=color, linewidth=1.4))
    if body:
        ax.text(x + width / 2, y + height * .72, title, ha='center', va='center',
                fontsize=size, color=color, weight='bold')
        ax.text(x + width / 2, y + height * .32, body, ha='center', va='center',
                fontsize=size - 1, color='#304c5c', linespacing=1.6)
    else:
        ax.text(x + width / 2, y + height / 2, title, ha='center', va='center',
                fontsize=size, color=color, weight='bold', linespacing=1.5)


def arrow(ax, start, end, label='', color=GRAY, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=16,
                                linewidth=1.5, color=color, linestyle='--' if dashed else '-'))
    if label:
        x = (start[0] + end[0]) / 2
        y = (start[1] + end[1]) / 2
        ax.text(x, y + .035, label, ha='center', fontsize=10.5, color=color,
                bbox=dict(facecolor='white', edgecolor='none', pad=1))


def save(fig, name, footnote):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.text(.035, .035, footnote, fontsize=10.5, color=GRAY, linespacing=1.5)
    for suffix in ('png', 'svg'):
        target = OUT / f'{name}.{suffix}'
        options = {'metadata': {'Date': None}} if suffix == 'svg' else {}
        fig.savefig(target, dpi=160, facecolor='white', **options)
        if suffix == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text(encoding='utf-8').splitlines()) + '\n',
                              encoding='utf-8', newline='\n')
    plt.close(fig)


def overview():
    fig, ax = canvas('同一 backend 内的请求路径与状态归属', 'lookup 移除后的实现结构，箭头表示数据或依赖关系')
    box(ax, .015, .74, .19, .18, 'HTTP 请求', '文本 + 参考音频')
    box(ax, .29, .74, .27, .18, '准备线程池', '相同参考音频共享 future')
    box(ax, .67, .74, .31, .18, 'prompt_stream', '准备说话人 / Flow 初始状态')
    arrow(ax, (.21, .83), (.28, .83))
    arrow(ax, (.57, .83), (.66, .83))
    box(ax, .015, .34, .27, .23, 'NPU 主线程', 'Thinker 普通 decode\nTalker K10 设备块')
    box(ax, .37, .34, .22, .23, 'chunk25 窗口队列', '请求 ID / codec / last')
    box(ax, .67, .34, .31, .23, '音频线程 / t2w_stream', '批量 Euler Flow + HiFT\n每个请求持有流式 cache')
    arrow(ax, (.11, .73), (.11, .58), '文本')
    arrow(ax, (.29, .455), (.36, .455))
    arrow(ax, (.60, .455), (.66, .455))
    arrow(ax, (.825, .73), (.825, .58), 'adopt prompt')
    box(ax, .015, .015, .27, .17, 'Thinker / Talker KV 页池', '页表与逻辑长度按请求管理', GREEN, '#edf6ef')
    arrow(ax, (.15, .33), (.15, .195))
    box(ax, .67, .015, .31, .17, 'HTTP 流式响应', '24 kHz PCM16 + sample offset', ORANGE, '#fcf2e9')
    arrow(ax, (.825, .33), (.825, .195))
    ax.text(.475, .09, '普通 launch 由共享 gate 协调\ncapture 排他，媒体 scratch 由锁保护',
            ha='center', va='center', color=GRAY, fontsize=11.5, linespacing=1.6)
    save(fig, 'runtime-overview', '结构本身提供减少跨 stage 边界的机会，独立 IPC / context 成本尚无消融百分比。')


def prompt_lookup():
    fig, ax = canvas('Prompt-lookup 的一次拒绝示例', '数字为示意 token ID，原 Thinker 对草稿进行因果块验证', 8.3)
    rows = [('本请求 prompt', [11, 22, 33, 44, 55], .81),
            ('已有生成', [11, 22], .63),
            ('草稿提案', [33, 44, 55], .45),
            ('原模型预测', [33, 99, 55, 77], .27),
            ('本轮接收', [33, 99], .09)]
    for label, tokens, y in rows:
        ax.text(.01, y + .055, label, ha='left', va='center', color=GRAY, fontsize=13)
        for index, token in enumerate(tokens):
            color, fill = BLUE, '#edf5f9'
            if label == '草稿提案' and index >= 1:
                color, fill = '#ac544b', '#fcefeb'
            if label == '原模型预测' and index >= 2:
                color, fill = '#87939b', '#f0f2f3'
            if label == '本轮接收' or (label == '原模型预测' and index == 0):
                color, fill = GREEN, '#e9f5ec'
            box(ax, .20 + index * .108, y, .082, .11, str(token), color=color, fill=fill, size=17)
    box(ax, .785, .11, .195, .77, '验证与 KV',
        '验证输入\n[22, 33, 44, 55]\n\n草稿 44 首先不匹配\n接收 target 99\n\nKV 追加保留输入\n[22, 33]',
        color=ORANGE, fill='#fcf2e9', size=12)
    save(fig, 'prompt-lookup-example', '验证前逻辑 KV 长度为 L，写入 4 个输入位置后保留 L + 2。下一步将 99 作为待输入的尾 token。')


def talker():
    fig, ax = canvas('Talker 从逐帧主机控制到 K10 完整设备块', '帧之间仍保持因果依赖，图重放减少主机逐帧派发')
    ax.text(.01, .92, '逐帧主机控制示意', color=GRAY, fontsize=13, weight='bold')
    for index in range(10):
        x = .015 + index * .097
        box(ax, x, .71, .077, .13, f'frame\n{index + 1}', color=GRAY, fill='#f0f3f5', size=11)
        if index < 9:
            arrow(ax, (x + .079, .775), (x + .095, .775))
    box(ax, .015, .35, .19, .22, 'Host bind', 'KV 页表 / position\nseed / history')
    box(ax, .29, .35, .40, .22, '一次 NPUGraph replay',
        'frame 1 → … → frame 10\nembedding / decoder / head / sampling / state')
    box(ax, .78, .35, .20, .22, '块 token 回读', '按原顺序处理\nEOS / 文本块边界', GREEN, '#edf6ef')
    arrow(ax, (.21, .46), (.28, .46))
    arrow(ax, (.70, .46), (.77, .46))
    box(ax, .015, .025, .28, .15, '11 请求重分片为 7 + 4', color=ORANGE, fill='#fcf2e9', size=13)
    arrow(ax, (.30, .10), (.38, .10))
    box(ax, .39, .025, .24, .15, 'graph(7, K10)', color=ORANGE, fill='#fcf2e9', size=13)
    box(ax, .70, .025, .28, .15, 'graph(4, K10)', color=ORANGE, fill='#fcf2e9', size=13)
    arrow(ax, (.64, .10), (.69, .10))
    save(fig, 'talker-graph-example', '当前整块图覆盖请求行数 1 / 2 / 4 / 5 / 6 / 7 / 8。容量与状态条件不满足时走常规 decoder / graph-lane 路径。')


def host_lengths():
    fig, ax = canvas('把已知累计 query 长度保留在 host', '相同的布局信息同时服务 device 计算与 host attention 参数构造')
    ax.text(.01, .92, '旧路径', color=GRAY, fontsize=13, weight='bold')
    box(ax, .015, .64, .22, .20, 'layout builder', 'new_tokens = [2, 1, 3]')
    box(ax, .30, .64, .22, .20, 'NPU qo_indptr', '[0, 2, 3, 6]')
    box(ax, .59, .64, .18, .20, '每层 D2H 观察', '[2, 3, 6]', ORANGE, '#fcf2e9')
    box(ax, .84, .64, .14, .20, 'attention')
    arrow(ax, (.24, .74), (.29, .74))
    arrow(ax, (.53, .74), (.58, .74))
    arrow(ax, (.78, .74), (.83, .74))
    ax.text(.01, .46, 'r337 路径', color=GREEN, fontsize=13, weight='bold')
    box(ax, .015, .14, .22, .20, 'layout builder', 'new_tokens = [2, 1, 3]', GREEN, '#edf6ef')
    box(ax, .30, .14, .22, .20, 'host_qo_indptr', '(0, 2, 3, 6)', GREEN, '#edf6ef')
    box(ax, .59, .14, .18, .20, '校验并取末端', '[2, 3, 6]', GREEN, '#edf6ef')
    box(ax, .84, .14, .14, .20, 'attention', color=GREEN, fill='#edf6ef')
    arrow(ax, (.24, .24), (.29, .24))
    arrow(ax, (.53, .24), (.58, .24))
    arrow(ax, (.78, .24), (.83, .24))
    arrow(ax, (.68, .35), (.68, .63), color='#ad594b', dashed=True)
    ax.text(.75, .49, 'host 元数据缺失\n或校验不满足', ha='left', va='center', color='#ad594b', fontsize=11)
    save(fig, 'host-lengths-example', '布局仍生成 device qo_indptr。host 副本来自同一个 builder，验证行数、起点、总 query 行数与每段增量。')


def warmup():
    fig, ax = canvas('在真实执行者上完成媒体图预热', '初始化依赖示意，health 就绪前完成对应 worker / stream 的通用桶 capture')
    titles = ['配置读取', '音频 worker', 'ready 事件', '准备线程池', 'health 就绪']
    bodies = ['rows 1/2/4/8\nextended 开启',
              't2w_stream\nEuler + HiFT capture',
              '构造线程等待\n传播初始化错误',
              'prompt_stream\nsetup Euler capture',
              '真实请求进入\n已有 shape replay']
    for index, (title, body) in enumerate(zip(titles, bodies)):
        x = .015 + index * .199
        box(ax, x, .43, .17, .33, title, body, size=12)
        if index < 4:
            arrow(ax, (x + .173, .595), (x + .191, .595))
    box(ax, .015, .06, .965, .20, '固定零输入预热，实际请求重绑自己的输入与 cache',
        '普通执行遵守 graph launch gate，prompt 与 audio 继续由共享 scratch 锁协调',
        GREEN, '#edf6ef', size=13)
    save(fig, 'media-owner-warmup', '预热增加初始化时间和图内存。r337 音频 worker 媒体图初始化约 6.70 秒，prompt setup 与其他模型初始化另计。')


if __name__ == '__main__':
    overview()
    prompt_lookup()
    talker()
    host_lengths()
    warmup()
    print('Rendered five explanatory diagrams as SVG and PNG')
