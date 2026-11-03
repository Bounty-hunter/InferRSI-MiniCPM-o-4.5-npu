"""Direct NPU product comparison with one aggregated candidate curve."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from r337_comparison_data import CONCURRENCIES, REQUEST_COUNTS, load_direct_comparison

HERE=Path(__file__).resolve().parents[1]/'evidence/20261103-r337/campaign'
R=HERE/'results'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
assert read(HERE/'omni-confirmation-verification.json')['verified']
cs=list(CONCURRENCIES)
omni,engine,paths=load_direct_comparison()
fields=('mean_audio_ttfp_ms','mean_audio_rtf','request_throughput',
        'p99_audio_ttfp_ms','p99_audio_rtf','total_audio_duration_s')
engine_result={c:{key:sum(engine[r][c][key] for r in (1,2))/2 for key in fields} for c in cs}

font=FontProperties(family=['Microsoft YaHei','Noto Sans CJK SC','WenQuanYi Micro Hei','SimHei'])
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,
                     'font.size':11,'svg.fonttype':'path','pdf.fonttype':42})
ORANGE='#cb632b';BLUE='#126b9a';INK='#183545'
FOOT=('引擎每档指标取两次测量的算术平均。橙线为 Omni 基线，蓝线为独立引擎 r337。\n'
      'Flow 引擎 FP16 / 6步，Omni FP32 / 10步。RNG实现及生成音频时长不同。\n'
      'C1 为后补测量。C2/C4/C8 沿用此前完成的同配置测量。\n'
      '完整语音质量门待完成，评估迁至 Omni NPU 环境。')

def base(titles,title):
    fig,axs=plt.subplots(1,len(titles),figsize=(15.6,6.7))
    axs=np.atleast_1d(axs)
    for ax,t in zip(axs,titles):
        ax.set_title(t,fontsize=13.5,color=INK,pad=16)
        ax.set_xscale('log',base=2)
        ax.set_xticks(cs,[f'C{c}' for c in cs]);ax.set_xlim(.72,10.5)
        ax.minorticks_off()
        ax.set_xlabel('并发数',labelpad=10)
        ax.grid(axis='both',alpha=.16);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['bottom','left']].set_color('#a9b3b8')
        ax.tick_params(colors='#3b505c')
        ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle(title,fontsize=23,color=INK,y=.97,weight='bold')
    fig.text(.5,.902,'单卡 Ascend 910B1（A2）｜官方客户端 5917730a｜chunk25｜C1/C2/C4/C8 = 32/32/64/128 请求',
             ha='center',fontsize=11,color='#526671')
    fig.text(.5,.035,FOOT,ha='center',va='bottom',fontsize=9.5,color='#53626b',linespacing=1.55)
    fig.subplots_adjust(left=.06,right=.98,top=.735,bottom=.29,wspace=.28)
    return fig,axs

def series(ax,key,gain=False):
    a=np.array([omni[c][key] for c in cs])
    b=np.array([engine_result[c][key] for c in cs])
    if gain:
        b=100*(b/a-1) if key=='request_throughput' else 100*(1-b/a)
        ax.axhline(0,color=ORANGE,lw=1.5,ls=':',label='Omni 基线（0%）')
    else:
        ax.plot(cs,a,'o-',color=ORANGE,lw=2.5,ms=7,label='vLLM-Omni main@9990f10d9')
    ax.plot(cs,b,'s-',color=BLUE,lw=2.3,ms=6,label='独立引擎 r337')
    return a,b

def save(fig,stem):
    for ext in ('png','svg','pdf'):
        fig.savefig(HERE/f'{stem}.{ext}',dpi=180,facecolor='white')
    plt.close(fig)

def legend(fig,ax):
    hs,ls=ax.get_legend_handles_labels()
    fig.legend(hs,ls,loc='upper center',bbox_to_anchor=(.5,.866),ncol=2,
               frameon=False,fontsize=10.5,handlelength=3,columnspacing=2)

metrics=[('mean_audio_ttfp_ms','TTFP 均值（ms，越低越好）',lambda v:f'{v:.0f}'),
         ('mean_audio_rtf','音频 RTF 均值（越低越好）',lambda v:f'{v:.3f}'),
         ('request_throughput','request/s（越高越好）',lambda v:f'{v:.3f}')]
fig,axs=base([m[1] for m in metrics],'独立引擎 vs vLLM-Omni · 原生均值')
for ax,(key,_,fmt) in zip(axs,metrics):
    a,b=series(ax,key)
    ax.set_ylim(0,max(a.max(),b.max())*1.24)
    higher=key=='request_throughput'
    for i,c in enumerate(cs):
        ax.annotate(fmt(a[i]),(c,a[i]),xytext=(0,-19 if higher else 11),textcoords='offset points',
                    ha='center',fontsize=10,color=ORANGE)
        ax.annotate(fmt(b[i]),(c,b[i]),
                    xytext=(0,12 if higher else -21),textcoords='offset points',ha='center',
                    fontsize=10,color=BLUE)
    if key=='mean_audio_rtf':
        ax.axhline(1,color='#9aa3aa',lw=1,ls=':',zorder=0)
        ax.text(1.05,1.025,'RTF = 1',color='#7c878e',fontsize=9)
legend(fig,axs[0]);save(fig,'独立引擎vs-vLLM-Omni-均值')

fig,axs=base(['TTFP 均值降低（越高越好）','RTF 均值降低（越高越好）',
              'request/s 增加（越高越好）'],'独立引擎相对 vLLM-Omni · 原始性能差值')
for ax,(key,_,_) in zip(axs,metrics):
    _,b=series(ax,key,gain=True)
    ax.set_ylim(min(0,b.min()*1.28),b.max()*1.28)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100,decimals=0))
    for i,c in enumerate(cs):
        ax.annotate(f'{b[i]:.1f}%',(c,b[i]),xytext=(0,12),textcoords='offset points',
                    ha='center',fontsize=10,color=BLUE)
legend(fig,axs[0]);save(fig,'独立引擎vs-vLLM-Omni-性能差值')

fig,axs=base(['TTFP P99（ms，越低越好）','音频 RTF P99（越低越好）'],
             '独立引擎 vs vLLM-Omni · 样本 P99')
fig.text(.5,.198,'引擎取两次运行 P99 的算术平均。每次各档32/32/64/128个请求，P99作尾部诊断参考。',
         ha='center',fontsize=10,color='#526671')
for ax,key in zip(axs,('p99_audio_ttfp_ms','p99_audio_rtf')):
    a,b=series(ax,key);ax.set_ylim(0,max(a.max(),b.max())*1.24)
    for i,c in enumerate(cs):
        fmt=(lambda v:f'{v:.1f}') if 'ttfp' in key else (lambda v:f'{v:.3f}')
        ax.annotate(fmt(a[i]),(c,a[i]),xytext=(0,12),textcoords='offset points',ha='center',color=ORANGE,fontsize=10)
        ax.annotate(fmt(b[i]),(c,b[i]),xytext=(0,-21),textcoords='offset points',ha='center',color=BLUE,fontsize=10)
legend(fig,axs[0]);save(fig,'独立引擎vs-vLLM-Omni-P99')

rows=[]
for c in cs:
    rows.append(dict(concurrency=c,measured_requests=REQUEST_COUNTS[c],supplementary_measurement=c==1,
        omni={k:omni[c][k] for k in fields},
        engine_result=engine_result[c],
        engine_repeats=[{k:engine[r][c][k] for k in fields} for r in (1,2)]))
manifest=dict(scope='Direct one-card NPU comparison by concurrency, not iteration number',
    candidate='r337',omni_server_sha=read(HERE/'omni-confirmation-verification.json')['spec']['server_sha'],
    concurrencies=cs,measured_requests=[REQUEST_COUNTS[c] for c in cs],warmups_per_tier=2,request_errors=0,
    c1_supplement=read(HERE/'c1-completion-verification.json'),
    sources=[dict(path=p.relative_to(HERE).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
    rows=rows,aggregation='Arithmetic mean of the two candidate run-level values for every metric and tier. Gains computed from the aggregated values against Omni.',
    p99_definition='Arithmetic mean of two run-level P99 values, not a pooled-request P99.',
    baseline_curves_shown=1,engine_curves_shown=1,engine_repeats_shown=0,
    candidate_source_runs_per_tier=2,speech_gate='NOT_RUN',equal_quality_claim=False,
    qualification='Omni Flow FP32/10 vs engine FP16/6; RNG and generated audio lengths differ; raw gains cannot all be assigned to RSI.',
    files=[f'{stem}.{ext}' for stem in ('独立引擎vs-vLLM-Omni-均值','独立引擎vs-vLLM-Omni-性能差值','独立引擎vs-vLLM-Omni-P99') for ext in ('png','svg','pdf')])
(HERE/'独立引擎vs-vLLM-Omni-绘图数据.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Generated direct mean, gain and P99 curves: PNG / SVG / PDF, with source-bound data')
