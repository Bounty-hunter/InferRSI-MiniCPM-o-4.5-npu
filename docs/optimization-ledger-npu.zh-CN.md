# NPU 优化完整账本：r32–r306

当前r337候选、C1/C2/C4/C8单结果曲线及后续40轮记录已更新，详见[最新优化总结](rsi-r337-20261103.zh-CN.md)和[当前性能表](current-results.md)。下文保留相应阶段的历史证据。

本文保留早期NPU迁移、统一CLI活动、profiling及六层分析后的逐项记录。它是[完整历程](optimization-journey.zh-CN.md)的审计附录。
只列已有记录；编号包含准备、诊断、验证、重试和watcher，不能把最大编号当成算法优化次数。
原始文稿的章号7/8保留；其中“当前/默认/下一步”指当时工程状态，不是2026-11-02的发布状态。
GPU架构前身不作为本附录的NPU性能证据。

## 7. 早期 A2/NPU `r32–r181` 可核验账本

### 7.1 证据边界

这一段的 `rNN` 是远端作业/证据流水号，不等于每个数字都是一次算法优化。本地归档只覆盖一部分编号；因此表格列出所有能由本地 plan、config、manifest、完整报告或退出码证明的轮次，其他编号按区间统一标注“无本地可核验归档”。

该阶段先使用专用 K=5/K=10 verifier，指标是 `TTFA p50` 和 `frames/s`，验收目标为点估计和单侧 95% LCB 均达到 2×；后来才转向 Omni 官方 CLI 的 `TTFP/RTF/request/s`。旧结果不能直接与新 v0.32 活动横比。

### 7.2 r32–r78：冻结协议、运行时和第一版正式计划

| 轮次 | 来源 | 实际工作 | 结果/状态 | 决策 |
|---|---|---|---|---|
| r32 | 基线/工程 | 冻结 Ascend validator：固定 K=10、C1/C16/C32、TTFA p50、frames/s、错误率 0、语音质量门；性能 ratio 的 point 和单侧 95% LCB 均须 ≥2.0 | 验收协议落盘；不是性能候选 | 采用 |
| r36 | 质量工程 | 首次冻结完整 1088 条×seed 0/1/2、C1/C32 的 Omni reference 和 Q_ref 流程 | 建立绝对 floor＋相对 quality gate | 后续被更严谨身份绑定 supersede |
| r38 | 身份修复 | 将候选质量作业容器身份从错误的 `1033:1000` 修为当时核验的 `1033:1033` | 尚未启动候选质量 cell，故可无污染修订 | supersede r36 |
| r39 | workload 修复 | 强制候选与 Omni 使用逐 seed 完全相同的 closed-loop 请求顺序 | 消除请求顺序混杂变量 | supersede r38 |
| r41 | validator runtime | 将冻结性能 verifier 准备到远端运行时并做 self-check | 建立 runtime 身份链 | 被 r42 修订 |
| r42 | loader 修复 | verifier 从 experiment manifest 派生真实 Omni checkout，再导入 `bench.cli`；用真实完成 cell 预检 | 修复读取错误 checkout 的风险 | 采用 |
| r49 | 容器环境修复 | 为无 passwd 记录的 UID 注入 USER/LOGNAME/HOME、可写缓存，并启用 PyTorch device-backend autoload | 保持模型、请求、Q_ref 和阈值不变 | supersede r39 |
| r56 | 工程线索 | 本地历史只保留 workspace/cache 修复线索，未找到独立正式报告 | 无可核验性能数字 | 不宣称收益 |
| r69 | 引擎配置 | 形成 C1/C16/C32 Ascend 专用配置快照 | 有配置，无本地完整正式报告 | 作为 r71 前置 |
| r71 | 性能计划 | profiler-free final source，K=5，C1/C16/C32 候选与 Omni 成对比较 | 计划和配置可核验；本地未保留完整 gate 报告 | 不外推数字 |
| r76 | 质量绑定 | 将候选质量生成绑定到 profiler-free r71 源码和并发专属配置 | 修复质量候选与性能候选可能不是同一源码的问题 | 采用，后续再被 r137 supersede |
| r78 | 性能计划 | 固定 K=10，在同一活动中安排 C1/C16/C32 baseline/candidate 六格 | 计划完整；本地没有六格最终冻结报告 | 进入后续调优 |

### 7.3 r84–r109：C16 调度、尾批和 prompt overlap

| 轮次 | 方案 | 可核验服务结果 | 解释/决策 |
|---|---|---|---|
| r84 | r71 配置 C16 探针 | 320/320 失败，`valid=false`；错误结果里的 req/s 无意义 | 失败轮，不采用 |
| r85 | workspace-share 修复/探针 | 320 请求、0 失败；`136.90 frames/s`，TTFA p50/p95 `7.971/13.733 s`，RTF `2.8634`，`1.1162 req/s` | 恢复有效服务，作为调度对照 |
| r86 | steady batch wait=50 ms 配置 | 只有配置归档，无本地完整结果 | 不宣称收益 |
| r87 | tail-batch 探针 | 0 失败；`140.64 frames/s`，TTFA p50/p95 `8.360/9.579 s`，RTF `2.7847`，`1.1360 req/s` | 吞吐略升、p95 显著收敛，p50 变慢；继续拆尾批 |
| r88 | tail batch wait=20 ms | 只有配置归档 | 无正式收益结论 |
| r90 | tail batch wait=5 ms | 只有配置归档 | 无正式收益结论 |
| r92 | tail min-ready=4 | 只有配置归档 | 无正式收益结论 |
| r93 | min-ready=4、wait=50 ms | 0 失败；`146.57 frames/s`，TTFA p50/p95 `7.973/11.813 s`，RTF `2.6715`，`1.1915 req/s` | 比 r85 吞吐和 RTF 更好，但不是冻结 K=10 gate |
| r94 | min-ready=4、wait=100 ms | 只有配置归档 | 无正式收益结论 |
| r101 | prompt overlap、min-ready=4、wait=10 ms | 0 失败；`143.62 frames/s`，TTFA p50/p95 `7.945/12.926 s`，RTF `2.7275`，`1.1740 req/s` | 未超过 r93 吞吐/RTF；不视为冠军 |
| r104 | prompt overlap＋prep2 | 只有配置归档 | 无本地正式服务结果 |
| r106 | prompt setup batching | 只有配置归档 | 无本地正式服务结果 |
| r109 | prompt overlap＋prep2＋affinity | 只有配置归档 | 无本地正式服务结果 |

### 7.4 r113–r131：静态 FIA、Talker post-graph 和 decoder-op 消融

| 轮次 | 来源/改动 | 可核验结果 | 决策 |
|---|---|---|---|
| r113 | C1 static FIA capacity=512 | 仅配置归档 | 不宣称收益 |
| r114 | C1 static FIA capacity=256 | 仅配置归档 | 进入 r117 小样本组合 |
| r115 | static FIA256、K=25 | 仅配置归档 | 后续验收回到冻结 K=10 |
| r117 | 单请求消融：device block、FIA256、post-graph | 基础 block：TTFA `0.9189 s`、`2.449 audio-s/s`；FIA256：`0.7914 s`、`3.045`；post-graph：`0.7825 s`、`2.876`；均 1/1 成功 | 只算 L1 小样本，不能当正式服务收益 |
| r118 | FIA256＋post-graph 修复组合 | 单请求 TTFA `0.7039 s`、`3.419 audio-s/s`，1/1 成功 | 解锁更大范围验证 |
| r119 | static FIA256 exact | 仅配置归档 | 无独立服务结论 |
| r120 | static FIA256＋Omni sampling | 仅配置归档 | 为采样语义对齐做准备 |
| r125 | Omni native RNG＋合格 decoder ops | C1/C16/C32 配置齐全；成为 r135/r136 及 r137 quality 的候选身份 | 保留为旧活动正式候选 |
| r127 | unfused eager outlier 对照 | 4/4 成功；TTFA `2.001 s`，`0.942 audio-s/s` | 建立 eager 控制 |
| r128a | unfused graph | 4/4；TTFA `0.843 s`，`2.941 audio-s/s` | graph 对小样本明显有效，但不是冻结服务 gate |
| r128b | fused eager | 4/4；TTFA `1.467 s`，`1.313 audio-s/s` | 融合改善 eager，但仍慢于 graph |
| r129 | Talker decoder-op 消融：reference/RMS/SwiGLU/Add+RMS/full；Thinker 固定 full | 4/4 均成功；TTFA `0.762–0.784 s`；audio-s/s：reference `3.058`、RMS `3.303`、SwiGLU `3.124`、Add+RMS `3.173`、full `3.477` | full 在该微型 screen 吞吐最高；只作为候选选择依据 |
| r130 | Thinker decoder-op 消融；Talker 固定 reference | 4/4 均成功；TTFA `0.795–0.857 s`；audio-s/s：reference `2.919`、RMS `2.902`、SwiGLU `3.107`、Add+RMS `2.990`、full `3.005` | SwiGLU 局部最好，但差异未经过正式统计门 |
| r131 | Thinker Add+RMS＋Talker Add+RMS 组合 | 4/4；TTFA `0.823 s`，`3.136 audio-s/s` | 未超过 r129 full 的局部吞吐；不作为冠军 |

### 7.5 r135–r153：正式六格、恢复作业、counter RNG 与旧冻结性能门

| 轮次 | 实际工作 | 结果/状态 | 决策 |
|---|---|---|---|
| r135 | 使用 qualified decoder ops 规划 K=10 C1/C16/C32 六格 | plan 可核验；本地未保留完整最终报告 | 由 r136 继续 |
| r136a–f | 在一张 A2 上运行 baseline/candidate 的 C1/C16/C32 六格 | 部分 cell 和前 4 个 run 已完成，r136a 后半未完整；不能拼成冻结结论 | 保留审计，不补猜 |
| r137 | 将完整 1088×3 seed、C1/C32 质量门绑定到 r125/r136 的实际源码和配置 | 质量 freeze 完成，但性能链随后换活动 | 没有最终 speech gate 结论 |
| r143a/r145a | 尝试只恢复 r136a 的 run4–run10，保留 run0–3 和污染审计 | 多次围绕环境 fingerprint、PYTHONPATH 和共享卡占用恢复；没有形成完整六格最终报告 | 旧活动停止扩展 |
| r146 | r136 恢复链的性能 watcher | 等待六格；未产生可用最终 gate | 归档 |
| r147a–f | 用户授权不复用旧部分数据，在等价 A2 卡上从零重采 K=10 六格 | 计划完整；后来用户改用官方 vLLM-Omni 1/4/8 benchmark，未把该活动作为现行结论 | 保留旧日志，不混入新基线 |
| r148 | r147 性能 gate watcher | 未形成现行冻结报告 | 归档 |
| r149 | 将质量 reference/candidate 绑定到 r147 新活动；补齐 quality runtime/pipeline/postprocess | 协议与工具冻结，但没有最终完整 speech gate | 归档 |
| r150 | Omni 过滤语义＋batched counter RNG，post-graph 关闭 | C16 screen：0 失败；`117.99 frames/s`，TTFA p50/p95 `9.866/11.695 s`，RTF `3.3142`，`0.9690 req/s` | 采样 launch 减少假设未在该 screen 获得端到端优势 |
| r151 | 在 r150 上开启 Talker head/filter/counter draw/state 的 post-graph | C16 screen：0 失败；`193.02 frames/s`，TTFA p50/p95 `8.801/10.299 s`，RTF `1.9889`；但仅 `0.6559 req/s`、makespan 487.9 s，说明 audio 长度/请求吞吐口径发生显著变化 | 进入 r152 正式成对验证；不能只看 frames/s |
| r152a–f | counter-postgraph K=10 C1/C16/C32 baseline/candidate 六格 | 成为 r153 verifier 的输入；0 error | 交给冻结门 |
| r153 | 冻结性能 verifier | C1：TTFA ratio point/LCB `2.152/2.142`，frames/s `2.975/2.796`；C16：`2.787/2.694`、`3.221/3.044`；C32：`4.398/4.105`、`4.820/4.454`；各 cell error rate=0，`performance_overall=pass` | **仅旧协议性能门通过**；报告明确 `speech_quality_evaluated=false`，不能视作最终验收 |

### 7.6 r164–r181：切换到 Omni 官方服务 benchmark、采样/graph parity 修复

| 轮次 | 实际工作 | 可核验结果 | 决策 |
|---|---|---|---|
| r164 | 尝试 Omni 官方基线 | exit 134 | 失败，不可比较 |
| r166 | Omni 官方 C1/C4/C8 基线 | 三个结果 JSON 已写出，但总作业 exit 134 | baseline 身份不完整；任何巨大收益只能标为 inconclusive |
| r167 | 专有候选首次官方 benchmark 尝试 | exit 3 | 失败 |
| r168 | 修订候选再次尝试 | exit 3 | 失败 |
| r169 | 候选完成 C1/C4/C8 | C1 `790.44 ms/0.2744/0.2455 req/s`；C4 `4356.85/1.6988/0.4944`；C8 `6239.90/2.0656/0.4388`；0 失败 | 高并发严重退化，继续诊断 |
| r170 | sampler diagnostic | eager uniform `65536/65536` 精确；graph uniform `65536/65536` 全不一致；postgraph vs eager token mismatch：rows1 `14/16`、rows4 `61/64`、rows8 `123/128` | 根因指向完整 RNG 状态/graph 语义 |
| r171 | graph 整数算子分解 | int32/int64 identity、shift、xor、multiply、mask 和 float conversion 均 graph=eager | 排除单一整数算子错误；继续修整体 counter 状态 |
| r172 | counter RNG L0 修复 | exit 0 | 通过最小门 |
| r173 | 固定 seed 实际模型 parity | C1/C4 共 8 个样本：text、Thinker token、codec token、frame/duration 全相同；post-graph replay 可见 | 通过模型级 parity |
| r174 | 修复后候选官方 benchmark | C1 `793.87/0.3230/0.6826`；C4 `4368.75/1.7893/0.4811`；C8 `6591.51/2.5270/0.7080`；0 失败 | C4/C8 仍慢，继续查 bucket/decoder parity |
| r175 | graph bucket parity | exit 1；C4 4/4 codec token 不同，C8 4/8 不同，部分 frame 数也变 | 失败，不能扩大 graph bucket |
| r177 | decoder parity 多 case | exit 1；128 measurements 中 21 个 hidden 非 exact，sampled mismatch 合计 17，另有 ring failure | 失败 |
| r178 | 缩小 decoder parity | exit 1，仍有 graph/ring failure | 失败 |
| r179 | 修订 parity harness/实现 | exit 0；在该轮冻结容差下通过，尽管内部不是所有 hidden/token byte-exact | 解锁服务复测，但不描述为逐位完全一致 |
| r180 | development speech content screen | 报告中的 graph1/graph8、C4/C8 样本 mean WER 均为 0，但总作业 exit 134，且 scope 明确不含 SIM/UTMOS/final gate | 只能保留内容筛查，不算质量通过 |
| r181 | 旧活动最终候选 | C1 `778.35 ms/0.3103/0.7064 req/s`；C4 `2031.19/0.7544/1.0122`；C8 `2911.41/1.0928/1.6646`；0 失败；内容筛查 WER=0 | 作为新活动的源码桥接快照；相对 r166 的百分比因 baseline exit 134 仅为 inconclusive |

### 7.7 未归档编号如何处理

- r32–r181 之间未在上述表中出现的流水号，本地没有足够证据判断它是代码修改、重跑、watcher、文件准备还是空档。
- 不把目录名或配置名推演成不存在的性能收益；只有完整 `summary.json`、正式 result JSON、report 或 exit-code 才写数值。
- r153 是“旧 K=10 verifier 性能通过”，r181 是“旧官方服务候选完成”，两者都没有构成当前发布结论：前者缺最终语音质量门，后者缺有效的旧 Omni 基线身份。
- 从 r182 开始重建 v0.32 Omni main 基线，才进入下述当前活动。

## 8. 新 v0.32 活动逐轮 RSI 实验账本

### 8.1 来源与状态口径

来源标记、知识条目和状态定义统一见第 4 节。以下保留实际完成的步骤；准备、诊断、watcher 和重跑不等同于算法优化轮。早期“保留/采用”仅表示进入后续工程实验，最终验收状态以第 1、10、11 节为准。

### 8.2 r181–r195：环境、基线与第一轮反例

| 轮次 | 来源 | 本轮做了什么 | 结果/收益 | 决策 |
|---|---|---|---|---|
| r181 | 基线/工程 | 冻结当时专有引擎源码快照，作为后续所有单变量候选的共同起点 | 只建立代码身份；无性能结论 | 保留基线 |
| r182 | 基线/工程 | 在新镜像 `9fd63cd20600`、vLLM 0.32 环境建立隔离 workspace 和首个 Omni main 作业 | 首次作业因环境/依赖未完成而失败；无有效 benchmark | 修依赖后重跑 |
| r183 | 基线/工程 | 修复依赖后重试 Omni 官方基线 | 仍因 NPU 原生算子兼容问题退出；无有效结果 | 用最小算子探针定位 |
| r184 | 现场证据 | 单独探测新镜像内所需 NPU 原生算子和调用方式 | 探针通过，证明不是模型/数据本身不可运行 | 为 r185 解锁 |
| r185 | 基线/工程 | 使用官方 CLI 跑 Seed-TTS C1/C4/C8 基线 | Omni：C1 `860.24/0.3115/0.8022`；C4 `1507.57/0.6567/1.4700`；C8 `2177.56/1.1690/1.7300`；0 错误 | 冻结为正式对照 |
| r186 | 基线/工程 | 首次把 r181 专有引擎接入相同官方客户端 | 健康检查把纯文本当 JSON 解析，作业失败；没有正式样本 | 修 harness，不改引擎 |
| r187 | 基线/工程 | 修正健康检查后完成初始专有引擎 C1/C4/C8 | C1 `782.50/0.3261/0.7325`；C4 `2034.19/0.7773/1.2040`；C8 `3097.81/1.1883/1.6063`。除 C1 TTFP 外基本落后 Omni | 作为 NPU RSI 初始点 |
| r188 | 现场证据 | 尝试完整 NPU profiler，关联 CPU 栈、设备算子和请求标记 | trace 不完整/退出异常；不能发布气泡占比 | 结果仅审计，不作根因 |
| r189 | 现场证据 | 改成从 engine owner 线程挂 profiler block hooks | 仍非零退出，缺少完整 CPU↔NPU 关联 | 拒绝据此优化 |
| r190 | 现场证据 | 再改成线程级 profiler 接入 | 仍未获得可靠完整 trace | 停止 profiler 盲试 |
| r191 | 自主方案 | 消除 Talker 单行 K-block 中恒等索引 gather，先做 L0 | 首版 L0 失败；无服务结论 | 修正确性后再测 |
| r192 | 自主方案 | 修复恒等 gather 消融实现，覆盖 B1/B3/B4/B8、EOS、连续 block | 16 组 token/state 精确一致；局部 500 次约 `59.4→11.0 ms`，但只是微基准 | 允许进入服务 screen |
| r193 | 自主方案 | 正式跑恒等 gather 消融 C1/C4/C8 | 相对后续原代码对照 r194，吞吐约回退 2.6%/7.2%/2.6%；三档没有稳定收益 | 拒绝采用 |
| r194 | 基线/工程 | 立即回到未修改引擎做反向对照，排除时间漂移 | `791.11/0.3211/0.7228`、`1941.07/0.7697/1.1954`、`3062.65/1.1923/1.5986` | 证明 r193 负收益 |
| r195 | 现场证据 | 使用引擎已有同步阶段计时跑 C4 前 8 条 | Thinker 累计约 7.90 s、Talker 4.98 s、Flow 7.96 s、其中 solve 6.88 s；线程有重叠，不能相加 | 将假设转向整块设备执行 |

### 8.3 r196–r215：Talker 完整 K-block 路线

| 轮次 | 来源 | 本轮做了什么 | 结果/收益 | 决策 |
|---|---|---|---|---|
| r196 | 现场证据＋自主方案 | 尝试 Thinker 静态 FIA/graph 路径，要求文本、Thinker token、codec 与原路径复现 | parity 未过，作业 exit 1 | 不进入性能测试 |
| r197 | 基线/工程 | 根据 r196 失败准备转向 Talker full-block 的实现与测试文件 | 没有独立启动的候选或性能样本 | 由 r198 接续 |
| r198 | KB→适配＋自主方案 | 把 Talker embedding、decoder、projection、sampling、state update 组成完整 K-block 图；先跑基础 L0 | 基础函数检查通过；无服务收益结论 | 继续补边界 |
| r199 | 自主验证 | 增加真实权重、EOS、连续 block 边界检查 | 24 组实际权重＋8 组 EOS 检查通过 | 进入整模型验证 |
| r200 | 自主验证 | 用实际模型做完整 Talker block L2 | 通过；证明 block 不是只对 stub 成立 | 允许服务 screen |
| r201 | 自主方案 | 第一版 full-block 官方 C1/C4/C8 | C1 `756.10/0.3091/0.7504`；C4 `2087.88/0.8113/1.1598`；C8 `3257.66/1.2118/1.5357`。C1 改善，C4/C8 仍明显差 | 暂不采用，继续查构图/状态 |
| r202 | 自主验证 | 尝试 warmed full-graph 整模型 A/B/A | 检查失败，暴露首次 capture、cohort 和状态轨迹未严格对齐 | 不把失败归因给 graph 数值 |
| r204 | 自主验证 | 用独立 capture stream 复测完整模型 | 仍出现整模型差异；无性能结论 | 继续拆 metadata/状态所有权 |
| r206 | 自主方案 | 把 position/slot/length 等逐帧整数 metadata 按原规则留在图外，只将 tensor copy/use 放图内 | 验证暴露物理页重绑定场景，未直接进入服务 | 增加 rebind 测试 |
| r207 | 自主验证 | 条件式 rebind L0 作业 | 前置状态未满足，未形成新的正式结果 | 由 r208 诊断替代 |
| r208 | 自主验证 | 专门测试碎片化物理页重新绑定和 EOS | 8 组 rebind＋8 组 EOS 精确通过 | 修复一个正确性盲区 |
| r210 | 自主验证 | 做整模型 cohort A/B/A | 失败；当时 warm/full 两边的 batch 轨迹不一致 | 追踪 Thinker 条件来源 |
| r212 | 现场证据 | 在进入 Talker 前记录 condition hash 和 batch 轨迹 | 发现差异已经在 Talker 之前产生，不能怪 graph | 根因转向 admission 门控 |
| r214 | 自主方案 | 改为收齐、排序完整 cohort 后统一 admission，再做 A/B/A | 5 组完整模型 token、条件、codec、帧数一致 | 通过严格 L2 |
| r215 | 自主方案 | 用修正后的 warmed K=10 block 跑正式 C1/C4/C8 | 相对 r194：TTFP +3.5/+3.1/+4.8%，RTF +1.8/+2.9/+3.2%，RPS +2.0/+3.7/+4.5%；相对 Omni 仍未全面领先 | 保留为后续控制基线 |

注：`r203/r205/r209/r211/r213` 没有独立正式候选，它们是文件准备、失败后的代码修订或编号空档，因此没有可单列的性能结果。

### 8.4 r216–r226：InferRSI 算子知识迁移与质量否决

| 轮次 | 来源 | 本轮做了什么 | 结果/收益 | 决策 |
|---|---|---|---|---|
| r216 | KB→适配 | Talker Add+RMS 单因素融合；固定 4 条 Seed-TTS、fresh-process A/B/A、冻结 evaluator | 开发筛查通过：WER=0，SIM/UTMOS 和静音、重复、截断均在门内；不是完整 speech gate | 进入性能 screen |
| r217 | KB→适配 | Add+RMS 正式 C1/C4/C8 | 相对 r215：TTFP +1.72/+3.57/-1.90%，RTF +2.60/+3.26/+0.46%，RPS +3.35/+2.47/+1.31%；C8 TTFP 回退 | 有小收益但不作为通用冠军 |
| r218 | KB：固定调制量 | 将 Flow 每步只依赖固定 temb/权重的 adaLN 调制预计算并复用 | 真实权重下 mel/cache 逐位一致；微基准仅快 0.48%–1.76% | 收益太小，不优先进入正式服务 |
| r219 | 现场证据 | 最小单线程 Flow profiler | 段错误 exit 139，CPU FRAMEWORK 为空 | 只留失败证据 |
| r220 | KB→适配 | 对 r218 做固定样本内容/数值 screen | 正确性成立，但没有新增端到端性能证据 | 未推广 |
| r221 | KB 启发＋自主方案 | 将 causal Conv1d 原型改成窗口 GEMM，保留 cache 语义 | 16 组数值/cache 通过；C8 常见 shape 慢 8%–12% | 明确拒绝 |
| r222 | KB 融合方法＋自主组合 | Thinker `full`：在 Add+RMS 上叠加原生 RMSNorm 和 SwiGLU，启动内容 screen | 原定卡队列未真正启动模型，后取消；没有结果 | 换空闲卡重跑 |
| r223 | 基线/工程 | 为 r222 准备替代调度/评分链 | 没有独立候选性能 | 由 r224 接续 |
| r224 | KB 融合方法 | 在空闲同型号卡完成 15 条 A/B/A 内容生成 | 文本/token 一致，condition/codec 可变化；生成 15/15 成功 | 进入性能和质量评分 |
| r225 | KB 融合方法 | Thinker full 正式 C1/C4/C8 | 相对 r215：C1 三项 +3.47/+4.05/+5.47%；C4 +1.68/+0.35/+0.79%；C8 TTFP -2.35%、RPS -1.07%；非全面收益 | 等待质量否决 |
| r226 | 冻结质量门 | 对 r224 实际 WAV 跑 UTMOS/SIM/WER | C1 UTMOS `4.34866→4.03321`，下降 0.31545；C4 SIM `47.25215→43.93563`，下降 3.31652；超门限 | 拒绝 r225，不采用 |

### 8.5 r227–r240：十轮 CFM/DiT RSI、稳定性修复与扩展性

| 轮次 | 来源 | 本轮做了什么 | 结果/收益 | 决策 |
|---|---|---|---|---|
| r227 | KB 融合知识＋自主组合 | 一次组合 packed QKV、AddLayerNorm、im2col causal conv、frames-major cache | FP16 CNN history 触发 Half/Float dtype 错误 | 拒绝首版 |
| r228 | 现场修复 | 修 dtype 后复验完整融合 | CNN cache max abs 0.0703、attention cache 0.0886，超过冻结数值门 | 拒绝完整融合 |
| r229 | 自主消融 | 分解 schedule-only、QKV、AddLayerNorm、conv、full | schedule-only 微基准 +12.74%，QKV +15.79%；AddLayerNorm/conv/full 虽快但 cache 数值失败 | 只保留 QKV 方向 |
| r230 | 自主验证 | packed QKV 覆盖 5 shape×2 输入 | 逐元素一致；图重放中位数 +15.59% | 进入语音筛选 |
| r231 | 自主验证 | QKV 候选做 A/B/A Thinker→condition→codec→WAV | 前三层一致，但 WAV 仅 1/5 byte exact；诊断发现误打包 conv 权重 | 修初始化，不做性能声明 |
| r232 | 现场修复 | `pack_convs` 只在 full variant 开启，QKV-only 不再污染 conv | 5 shape 再次逐元素一致，图重放中位数 +15.57% | 进入质量门 |
| r233 | 自主验证 | 修复后再做 A/B/A 内容筛查 | Thinker/condition/codec 一致；4/5 WAV 非 byte exact | 按用户授权交给质量门 |
| r234 | 基线/工程 | 打包冻结评分输入与证据身份 | 无独立性能结果 | 进入评分 |
| r235 | 冻结质量门 | 首次 QKV 评分作业 | 评分链准备/修订轮；不产生独立采用结论 | r236 给出正式筛查 |
| r236 | 冻结质量门 | 15 条、4 个唯一 row、seed0 的 UTMOS/SIM/WER | 所有绝对门和相对门通过；仍只是开发筛查 | 进入官方性能 |
| r237 | 正式性能 | packed-QKV 跑 C1/C4/C8 | C1 `705.87/0.2839/0.8269`；C4 `1725.90/0.6587/1.3968`；C8 NPU graph cache 崩溃，52/128 失败 | 可靠性不合格，拒绝 |
| r238 | 现场失败日志＋自主路由 | hybrid：rows<16 用 packed QKV，rows≥16 回退 schedule-only/native QKV | 5 shape 精确；rows8/16 交替 1024 次、102.25 s 无 507000 | 进入复测 |
| r239 | 正式性能 | hybrid 正式 C1/C4/C8 | `720.71/0.2906/0.8103`、`1797.96/0.7027/1.3123`、`2758.88/1.0938/1.7500`；0 错误。相对 r217 三档三指标均改善，但 C4 仍输 Omni | 作为稳定基线 |
| r240 | 用户问题＋现场证据 | 同 workload 扩到 C16/256、C32/512，对比 Omni | C16 `5992.92/2.2177/1.6429`，三项落后；C32 `12367.02/4.4625/1.5571`，相对 Omni +17.51/+9.57/+7.59% | 证明并发收益非单调 |

### 8.6 r241–r253：首包根因与选择性 Talker graph

| 轮次 | 来源 | 本轮做了什么 | 结果/收益 | 决策 |
|---|---|---|---|---|
| r241 | 现场证据 | 第一版 TTFP 时间线诊断，加入阶段标记和 fingerprint | 诊断产物/指纹需修订，不能作为最终因果表 | r242 重跑 |
| r242 | 现场证据 | 完成 C4/C16 首包分段时间线 | C16 相对 C4 新增 TTFP：prefill→首文本块 65.56%，首文本块→首 codec 20.15%，T2W queue 0.72% | 优先优化 Thinker/Talker tick |
| r243 | KB graph ownership＋自主方案 | Talker K=10 exact-batch graph 扩到 batch 3/5/6/7，做实际权重 parity | 420 个 codec token、KV、sampler state 精确一致 | 允许服务验证 |
| r244 | 自主验证 | 首次服务级 graph bucket parity | 首次 harness/门控不完整；无正式性能结论 | r244b 修订 |
| r244b | 自主验证 | 修正服务 cohort 后复测 graph bucket | 服务级 parity 通过 | 进入性能 |
| r245 | 自主方案 | batch 1–8 全部走 exact graph | C4 相对稳定版：TTFP -6.83%、RTF -5.75%、RPS -5.28%；C8/C16 有收益 | 一刀切策略拒绝 |
| r246 | 现场反馈＋自主路由 | 只启用 1/2/4/5/6/7/8；排除 row3；>8 有序分片，8+3 改 7+4 | 代码候选，不是独立测量轮 | 进入 r247 |
| r247 | 正式性能 | 选择性 graph 跑 C4/C8/C16 | 相对修改前：C4 TTFP -1.06%、RTF +0.01%、RPS +1.88%；C8 +5.99/+8.28/+9.59%；C16 +21.82/+21.84/+27.10% | 保留为通用性能候选 |
| r248 | 自主方案 | Thinker C16 decode 拆成 8+8 graph，做服务级 parity | 16/16 token 精确；小 cohort wall `14.49→8.11 s` | 进入正式服务，不先宣称收益 |
| r249 | 正式性能 | 8+8 microbatch 跑 C16/256 | 相对 r247：TTFP -9.37%、RTF -8.39%、RPS -7.72% | 拒绝 |
| r250 | 反事实验证 | 把 Thinker 1/2/4/8 graph 全部在启动阶段预捕获 | 预热约 1.59 s；用于排除 capture 污染 | 进入 r251 |
| r251 | 正式性能 | warmed Thinker graph 再跑 C16 | 相对 r247：TTFP -6.90%、RTF -6.58%、RPS -5.14% | 该 Thinker 预热未消除回退，拒绝此 8+8 方案 |
| r252 | 现场分析 | 汇总 graph 命中、fallback 和首包时间线 | 确认 row 形状覆盖及串行 tick 是主要中并发损失 | 形成差距报告 |
| r253 | 正式补测 | 对 r246 补 C1/C32，组成同一候选 C1–C32 表 | C1 三项 +17.67/+7.97/+2.21%；C32 +32.04/+26.94/+34.82%；0 错误 | 得到完整通用候选表 |

### 8.7 r254–r263：Thinker prefix KV 系列

| 轮次 | 来源 | 本轮做了什么 | C16 结果 `TTFP/RTF/RPS`，或其他结果 | 决策 |
|---|---|---|---|---|
| r254 | KB：固定前缀＋自主适配 | 对共享 ≥64-token Thinker 前缀建立 KV cache；先做 parity/probe，再跑 C4/C16 | C16 `3796.90/1.4480/2.5068`，相对 Omni -4.85/+30.75/+38.44%；但 C4 `1889.72/0.7158/1.3099`，三项回退 | C16 很强，不能全局启用 |
| r254b | 基线/工程 | 对 r254 probe/证据链进行补跑和审计 | 没有新的独立正式性能结论 | 保留审计 |
| r255 | 自主路由 | 仅 active≥8 时启用 prefix cache，试图保护低并发 | C16 `4207.42/1.5821/2.2855`；C4/C8 也未改善 | 阈值策略不理想 |
| r256 | 自主路由 | 把启用阈值提高到 active≥12 | C16 `4609.26/1.7004/2.1176`；C8 有 RTF/RPS 收益，C4 仍差 | 拒绝为通用方案 |
| r257 | 自主路由 | 改用 live-request 压力，`min_live=12` | C16 `4208.43/1.5966/2.2810` | 未超过 r254/r247 |
| r258 | 自主调参（有因果依据） | 将 `min_live` 从 12 降到 9，提高实际 cache 覆盖 | C16 `3867.70/1.5112/2.4028`；相对 Omni -6.81/+27.72/+32.69%，0 错误 | 保留为 C16 工程参照；非全档最终版 |
| r259 | 自主方案 | cache 一旦武装便 latch，避免压力波动反复开关 | `4166.81/1.5743/2.3102` | 比 r258 差，拒绝 |
| r260 | KB 生命周期＋自主方案 | 服务启动时 preseed 固定前缀，避免首批请求负责建 cache | `4135.38/1.5835/2.2935` | 未转化为端到端收益 |
| r260b | 基线/工程 | 在另一空闲同型号卡复核 preseed/部署链 | 无新的优胜结果 | 保留审计 |
| r261 | 自主方案 | preseed + latch + 10 ms arm wait，尝试合并首批请求 | `3912.12/1.5072/2.3909`；接近但没全面超过 r258 | 不替换 r258 |
| r262 | 自主方案 | cohort-aware admission，尝试让前缀复用批次更整齐 | `3979.84/1.5720/2.3213` | 等待成本抵消收益 |
| r263 | 自主反事实 | 把 arm wait 增到 50 ms，验证“再多等会是否更好” | `4086.25/1.5610/2.3450` | 更差；停止继续加等待 |
| r263b | 基线/工程 | 处理 r263 作业/证据收尾并复核退出状态 | 不产生新的性能候选 | 归档 r261–r263 |

### 8.8 r264–r276：Thinker 多步 device block

| 轮次 | 来源 | 本轮做了什么 | 结果/收益 | 决策 |
|---|---|---|---|---|
| r264 | 自主方案 | 用静态 FIA 把 Thinker 10 token 组成一个全图 block | 单步 0/16 token 差异，但 10 步后 13/160 token 不同，hidden L2=0.3017，KV 不精确 | 数值漂移，拒绝 |
| r264b | 自主诊断 | 逐步记录 hidden/cosine/KV，确认误差是否累积 | hidden cosine min 0.3309，证明不是偶发末步误差 | 终止静态 FIA |
| r265 | 自主方案 | 保留 dynamic paged attention 数值路径，把 10 步 host loop 变成 device block | batch16：160/160 token 精确、hidden L2=0、KV 精确 | 正确性通过 |
| r266 | 正式性能 | 只对 batch16 启用 device block 跑 C16 | `4144.36/1.5928/2.2675`，replay=0 | 实际流量未命中，不能归因收益 |
| r267 | 自主方案 | 将 device-block 支持扩成 batch range，并保留 eager fallback | 代码准备轮；无独立性能 | 进入阈值验证 |
| r268 | 正式性能 | 对 batch9–16 启用 block | `3953.74/1.5213/2.3779`，replay=0 | 仍未命中 |
| r269 | 自主验证 | batch8 实际权重 10-step parity | token/hidden/KV 精确 | 允许 r270 |
| r270 | 正式性能 | 只对 batch8 启用 block | `4097.58/1.5261/2.3845`，replay=0 | 未命中 |
| r271 | 自主验证 | batch3 实际权重 10-step parity | 30/30 token 精确、hidden L2=0、KV 精确 | 扩到小 batch |
| r272 | 正式性能 | batch3–8 block | `4164.29/1.5603/2.3513`；只命中 1 次 batch4；3 个 early EOS 正确处理 | 命中太少且更慢 |
| r273 | 自主验证 | batch1 实际权重 10-step parity | 10/10 token 精确、hidden L2=0、KV 精确 | 允许覆盖 1–8 |
| r274 | 正式性能 | batch1–8 block | `3930.46/1.4922/2.4133`；replay 仍为空。RTF/RPS略优于 r258、TTFP 更差，但机制零命中 | 不能把波动归因给 block |
| r275 | 现场计数器＋自主修复 | 找到 prefix 压力低于 9 时 `_thinker_step` 直接旁路 block；修成低压力且 prefill 完成时优先 block | 远端 7 tests passed；batch1/3/8/16 parity 已精确；尚无服务成绩 | 进入 r276 |
| r276 | 未完成验证 | 为 r275 路由准备 C16 runner，要求先证明 replay 非空 | 本地有准备文件，没有可用于本文的正式服务结果 | 保留备选，不是当前首要下一步 |

### 8.9 r277–r279：重新建立 CPU/NPU profiling 证据

| 轮次 | 实际工作 | 结果 | 决策 |
|---|---|---|---|
| r277/r277b/r277c | 用矩阵乘最小探针分离 profiler 故障 | `with_stack=True` 在 stop_trace exit139；CPU+NPU 不含原生栈可正常导出；原生 msprof 也可采集 | 使用原生事件＋补充 Python 栈，明确采样限制 |
| r278 | r258 稳定源码，官方 C4/64、C16/256 无 profiler 对照与 8 秒关联 trace | 0 失败、job exit0；控制 C4 `1809.26/0.7131/1.3217`，C16 `3598.06/1.3747/2.6319`；trace 解析强扰动，采集轮不算性能成绩 | 获得 CPU/NPU/调用栈线索，继续轻量复核 |
| r279 | 同一源码/配置，只记录主机时间戳和 attention 路径计数 | 64/256 正式请求，0 失败、job exit0；C4 混合 Thinker batch3 中位157ms，纯 decode37ms；C16 batch>8 占纯 decode 次数30.3%、wall time46.9% | 形成下一轮优化优先级；未实施新性能优化 |

r279 记录的 C16 admission→首音频路径均值约 3796ms，其中 Thinker prefill＋首文本块约59%，首 codec 窗口约15%，Token2Wav 排队＋处理约26%。这是 r258 配置在该轮的轻量时间线，替代不了客户端 TTFP，也说明旧 r242 的音频排队占比不应继续外推。原生 trace 空闲率有采集扰动，未作为可回收收益发布。

完整方法、逐项证据、候选方案和限制见：`r277–r279 profiling 分析`（原始审计文件）。本轮结束时新增服务和预约均已退出。

### 8.10 r280–r285：profiling 后的单变量验证

| 轮次 | 来源 | 修改/执行 | 真实结果 | 当前状态 |
|---|---|---|---|---|
| r280 | 现场 trace＋自主方案 | 混合 tick 按 decode→prefill 分组；首次服务 harness | 对照请求成功后 harness 读取错误字段，作业失败；不是有效完整 A/B | 修 harness，保留失败 |
| r280b | 单变量复测 | 同一源码重跑 C4 A/B/A、C16 B/A，开关分组策略 | C4 TTFP +5.84%、RTF +3.59%、吞吐 +4.27%；C16 −3.54%/−7.66%/−6.21%；0 请求错误，exit0 | 暂不采纳 |
| r281 | shape 频率＋自主图覆盖 | Thinker exact dynamic-PA graph 9/10；权重 parity 后 C16 A/B/A | batch9/10 token/hidden/KV 精确；原 B/A1 差值约19%/20%/24%，但 A2 184/256 失败，507000，exit1 | 原差值不作独立收益；保留失败 |
| r281b | 可靠性/热度复核 | 全新服务固定开启图，两次 C16/256 | B1 3915.45/1.53317/2.34864；B2 3432.34/1.31704/2.76256；512 条零错误，exit0 | 单独证明此次未复现错误 |
| r281c | 匹配对照 | 另一全新服务固定关闭图，两次同 workload | 冷窗口真实收益 +2.87%/+2.42%/+0.92%；热窗口 +2.55%/−0.28%/+1.00%；合计开关1024条零错误 | 候选待验收，非默认 |
| r282 | 冗余主机工作＋自主复用 | 发布校验后的 host Q metadata，复用 causal mask | 16 行 token/text 精确，命中数千次；C4 无一致收益，C16 吞吐仅 +0.07%；exit0 | 暂不采纳 |
| r283 | r280 回退反馈＋自主门控 | 连续繁忙周期 peak-live 达9后禁用低压力拆分，空闲才重置 | 22 单测、16 行 parity 通过；C4 +5.11%/+3.58%/+3.63%；C16 −8.13%/−12.09%/−9.00%；exit0 | 暂不采纳 |
| r284 | 冻结质量工具 | r283 48 WAV，16 行×control/candidate/control，绑定字节 hash | CUDA 无卡，3分钟排队超时、exit1；评分未启动 | 未完成，不能写质量 PASS/FAIL |
| r285 | 直接计时诊断 | 包装真实 scratch lock/capture context/请求标记，保留同步语义 | C16/256 零错误、exit0；首窗等锁48.40ms、进入解码前队列885.36ms；20次capture context并集23.60s | 诊断完成，未实施异步 D2H |

第 9 节给出最新实验的完整指标、修正结论与诊断范围。

### 8.11 如何阅读这份账本

- `r192/r218/r229/r230/r248` 的微基准或局部 wall time只能证明“值得进入下一层”，不能算最终收益。
- `r193/r221/r225/r245/r249/r251/r259/r260/r262/r263` 都是有价值的负结果：它们阻止了局部优化污染默认引擎。
- `r237` 性能很好但有 52 个错误，因此按可靠性门直接失败；不能挑完成的 76 条来宣传。
- `r254` C16 数字最好之一，但 C4 回退；`r258` 是当前更稳妥的 C16 screen；`r246` 才有同一候选的完整 C1–C32 表。
- 来源写为 KB，并不意味着代码从知识库复制或收益可继承；所有实现都必须重新经过本地 parity、官方服务和质量门。


## r286–r295：原生profiling、锁/capture、组合验证

本批包含诊断/失败/消融/验证，不是十项成功优化。下面早期screen与r286相比的数字不能代替r295匹配复测；
r292在C8/C32相对匹配r258回退，最终speech gate未通过，20秒候选补采失败。完整匹配原始数值见
[r295-matched-comparison.json](../evidence/optimization-journey/r295-matched-comparison.json)。

| 轮次 | 方案 / 来源 | 实际改动 | 实测 TTFP / RTF / req/s | 早期筛选相对r286收益 | 结论 |
|---|---|---|---|---|---|
| r286 | 锁与 capture 来源诊断；现场插桩 | 保持 r258，记录真实锁、capture 调用方、形状及请求阶段。 | 4014.06 / 1.56258 / 2.33296 | +0.00% / +0.00% / +0.00% | 诊断对照 |
| r287 | 公平 capture 门控；现场诊断→自主方案 | 等待 capture 时阻止新的非嵌套 reader；保留已有 reader 重入、nested capture 和异常恢复。 | 3738.24 / 1.40624 / 2.59648 | +6.87% / +10.01% / +11.30% | 保留供组合验证 |
| r288 | HiFT 通用启动预热；一次性工作前移→自主适配 | 使用零输入预热 rows 1/2/4/8/16 × frames 50/58；不消费 benchmark 请求或随机数。 | 3891.33 / 1.45817 / 2.49755 | +3.06% / +6.68% / +7.05% | 未选为最终候选 |
| r289 | Euler 通用启动预热；capture 来源证据→自主方案 | 独立测试五个通用 steady 形状，保留 eager fallback。 | 171成功/85失败；不列收益 | 不适用；验证另表 | 拒绝；不计算成功子集收益 |
| r290 | 公平门控＋HiFT 预热；组合消融 | 只组合已分别测过的两项机制。 | 3761.42 / 1.40756 / 2.58145 | +6.29% / +9.92% / +10.65% | 不叠加到选中候选 |
| r291 | 动态 PA task update 排他；失败日志→自主假设 | 在 _update_attention 周围使用 host launch 排他；不添加全局 device synchronize。 | 4387.62 / 1.62757 / 2.23729 | -9.31% / -4.16% / -4.10% | 拒绝 |
| r292 | 公平门控＋首包窗口优先；请求阶段证据→自主方案 | 对 ready 稳定排序，让 sample_offset==0 先于续包；保持请求绑定、消费次数及回退。 | 3597.84 / 1.36109 / 2.67127 | +10.37% / +12.89% / +14.50% | 选中作跨档验证，未正式采用 |
| r293 | 不可变 dense 页索引复用；重复元数据构造→自主方案 | 每请求按完整 physical-page tuple 缓存 page-ID tensor；增长、重绑、释放时安全失效。 | 3818.82 / 1.39856 / 2.57694 | +4.86% / +10.50% / +10.46% | 不纳入选中候选 |
| r294 | 匹配反向验证与 A/B/A 质量生成；验证 | 新进程对照/候选均按 C4→C16；另用三个新服务生成 control_before/candidate/control_after 各16条。 | 见匹配对照表；验证轮不是新算法 | 不适用；验证另表 | 保留验证证据，进入全档复测 |
| r295 | 统一五档验收与配对 profiling；验证与再诊断 | 两边各一个新服务，完全相同 C1→C4→C8→C16→C32 顺序；原生两条预热。另起新服务采集8秒配对trace，再补采20秒配对trace以覆盖更多请求。 | 见匹配对照表；验证轮不是新算法 | 不适用；验证另表 | 不采用；保留性能和诊断失败各自证据 |

## r296：六层诊断，不是新的优化候选

L0冻结口径；L1流水线；L2图覆盖/调度/设备间隙；L3冗余/布局；L4算子/计数器；L5传输/所有权；L6组件评估/验收。
真实NPU节点trace与后续host/客户端诊断分开；没有直接给profiler本身计算提速。

## r297–r306：六层分析后的十轮方案

以下native数值为TTFP mean ms / RTF mean / request/s；r306b修复第10轮API错误，不另计轮次。

| 轮次/补丁 | 来源 | 改动 | native初筛 | 决策 |
|---|---|---|---|---|
| [r297](../evidence/20261102-r305/patches/r297-vs-r292.patch) | 图批处理/请求状态知识库；r296大batch回退证据 | Thinker 图容量与 Talker 解耦，先覆盖16 | C16 3657.4/1.3908/2.6234 | 均值改善较小，未完成稳定性/质量验收 |
| [r298](../evidence/20261102-r305/patches/r298-vs-r292.patch) | r297筛选结果与C32覆盖缺口 | Thinker 图容量扩大到32，Talker仍为8 | C32 5877.4/2.1120/3.3289 | 均值改善，但P99回退，不直接采用 |
| [r299](../evidence/20261102-r305/patches/r299-vs-r292.patch) | 原创：源码显示整批Flow→整批vocoder→整批返回造成交付屏障 | T2W最多16行一组，每组完成立即交付PCM | C32 6392.3/2.3294/3.0050 | 高并发尾延迟候选；小并发不触发16行上限 |
| [r300](../evidence/20261102-r305/patches/r300-vs-r292.patch) | 对大图任务更新成本的反向假设 | Thinker纯decode拆为8行小批次走graph8 | C32 6740.1/2.4044/2.8956 | 均值、吞吐和播放连续性退步，拒绝 |
| [r301](../evidence/20261102-r305/patches/r301-vs-r292.patch) | r298均值/P99冲突；实际捕获耗时与缺少尾部桶 | 启动时预捕获Thinker 2/8/16/32桶 | C32 6191.7/2.2486/3.1181 | 启动成本/分桶/均值之间有权衡，未采用 |
| [r302](../evidence/20261102-r305/patches/r302-vs-r292.patch) | 源码过宽stream同步；设备气泡技能的依赖最小化原则 | 图元数据复用只等待上次replay事件 | C32 5813.3/2.1124/3.3330 | 增量均值很小，未证明稳定收益 |
| [r303](../evidence/20261102-r305/patches/r303-vs-r292.patch) | 已有AudioState/EDF；r299仍有续包迟到 | 16行交付分组内按播放deadline排序 | C32 6153.2/2.2757/3.0840 | 均值改善但相对r299尾部/1s缓冲断流更差，未采用 |
| [r304](../evidence/20261102-r305/patches/r304-vs-r292.patch) | L3不变量审查；原temb已固定但调制重复计算 | 固定timestep的CFM AdaLN调制表移出Euler replay | C16 4020.8/1.5058/2.4029 | 未获得稳定E2E收益；追加低并发复测亦未采用 |
| [r305](../evidence/20261102-r305/patches/r305-vs-r292.patch) | 首文本等待占主导；原创方案，借鉴有界重放/依赖闭包原则 | 通用prompt-lookup候选 + 原模型批量验证 + KV截断 | C2 902.8/0.4032/1.1820；C4 1389.7/0.6063/1.5302；C8 1893.0/0.8721/2.1997 | 重点性能候选；语音质量门未完成，未替换默认 |
| [r306](../evidence/20261102-r305/patches/r306-vs-r292.patch) | r296 .cpu主机等待；已有CUDA PendingAudio设计 | NPU pinned PCM + 完成事件，锁外消费并保护scratch依赖 | C2 1209.1/0.4794/1.0235；C4 1729.7/0.6743/1.3708；C8 2635.4/1.0060/1.8927 | 首试API错误已修复；低并发没有稳定收益，不采用 |

最终对Omni使用r305串行确认及C1补测，未从此表逐档挑最好值；详见[最新完整报告](rsi-profiling-20261102.zh-CN.md)。

## r307–r351后续演进

r307–r309补齐真实worker/stream上的媒体图预热，r310/r311为确认与控制。[三项方案和原生结果](../evidence/20261103-r337/followup/六层profiling继续优化-r307-r309.md)。

下面列出随后40个算法尝试。数值为TTFP mean ms / RTF mean / request/s，保留初筛观测与失败状态。最终r337使用独立正反序确认及C1补测。

| 轮次补丁 | 方案 | C2 | C4 | C8 | 初筛或失败结论 |
|---|---|---|---|---|---|
| [r312](../evidence/20261103-r337/campaign/patches/r312-vs-control309.patch) | thinker-init | 843.0/0.3817/1.2539 | 1224.1/0.5618/1.6135 | 1766.8/0.8186/2.3171 | no_consistent_gain_or_inconclusive |
| [r313](../evidence/20261103-r337/campaign/patches/r313-vs-control309.patch) | split-three | 773.5/0.3590/1.3149 | 1235.4/0.5534/1.6432 | 1757.7/0.7965/2.4033 | no_consistent_gain_or_inconclusive |
| [r314](../evidence/20261103-r337/campaign/patches/r314-vs-control309.patch) | draft4 | 934.7/0.3927/1.2144 | 1368.8/0.5917/1.5579 | 2038.2/0.8889/2.1539 | no_consistent_gain_or_inconclusive |
| [r315](../evidence/20261103-r337/campaign/patches/r315-vs-control309.patch) | unified-lookup | 773.4/0.3610/1.3336 | 1204.7/0.5452/1.7055 | 1708.3/0.7933/2.4228 | promising_needs_reverse_confirmation |
| [r316](../evidence/20261103-r337/campaign/patches/r316-vs-control309.patch) | batch-embedding | 810.2/0.3680/1.2875 | 1281.3/0.5571/1.6691 | 1768.4/0.8022/2.3893 | no_consistent_gain_or_inconclusive |
| [r317](../evidence/20261103-r337/campaign/patches/r317-vs-control309.patch) | bind-uploads | 812.5/0.3602/1.3220 | 1304.9/0.5715/1.6238 | 1949.8/0.9208/2.1102 | no_consistent_gain_or_inconclusive |
| [r318](../evidence/20261103-r337/campaign/patches/r318-vs-r312.patch) | tail-coverage | 812.9/0.3695/1.2866 | 1246.5/0.5676/1.6074 | 1673.8/0.7977/2.4076 | no_consistent_gain_or_inconclusive |
| [r319](../evidence/20261103-r337/campaign/patches/r319-vs-control309.patch) | host-lengths | 814.4/0.3722/1.2812 | 1184.7/0.5355/1.7226 | 1784.3/0.8315/2.2944 | no_consistent_gain_or_inconclusive |
| [r320](../evidence/20261103-r337/campaign/patches/r320-vs-control309.patch) | cores50 | 786.0/0.3344/1.4121 | 1153.5/0.5213/1.7860 | 1834.8/0.8679/2.2058 | no_consistent_gain_or_inconclusive |
| [r321](../evidence/20261103-r337/campaign/patches/r321-vs-control309.patch) | mask-cache | 784.1/0.3573/1.3316 | 1245.0/0.5603/1.6497 | 1766.0/0.8312/2.2933 | no_consistent_gain_or_inconclusive |
| [r322](../evidence/20261103-r337/campaign/patches/r322-vs-control309.patch) | cores75 | — | — | — | rejected_operator_resource_limit |
| [r323](../evidence/20261103-r337/campaign/patches/r323-vs-control309.patch) | draft16 | 802.4/0.3699/1.3063 | 1244.6/0.5783/1.6132 | 1814.5/0.8340/2.3192 | no_consistent_gain_or_inconclusive |
| [r324](../evidence/20261103-r337/campaign/patches/r324-vs-control309.patch) | unified-min5 | 782.8/0.3674/1.3091 | 1241.7/0.5605/1.6317 | 1734.2/0.8254/2.3089 | no_consistent_gain_or_inconclusive |
| [r325](../evidence/20261103-r337/campaign/patches/r325-vs-control309.patch) | page-id-cache | 813.3/0.3646/1.2944 | 1220.3/0.5469/1.6709 | 1727.5/0.8202/2.3543 | no_consistent_gain_or_inconclusive |
| [r326](../evidence/20261103-r337/campaign/patches/r326-vs-control309.patch) | hift-static-weightnorm | 810.1/0.3649/1.3204 | 1240.9/0.5421/1.6922 | 1773.0/0.8577/2.2280 | no_consistent_gain_or_inconclusive |
| [r327](../evidence/20261103-r337/campaign/patches/r327-vs-control309.patch) | private-prompt-flow | — | — | — | rejected_native_crash |
| [r328](../evidence/20261103-r337/campaign/patches/r328-vs-control309.patch) | packed-rotary | 865.3/0.3732/1.2786 | 1265.9/0.5672/1.6245 | 1848.9/0.8593/2.2368 | no_consistent_gain_or_inconclusive |
| [r329](../evidence/20261103-r337/campaign/patches/r329-vs-control309.patch) | cube50 | — | — | — | rejected_operator_resource_limit |
| [r330](../evidence/20261103-r337/campaign/patches/r330-vs-control309.patch) | vector50 | — | — | — | rejected_operator_resource_limit |
| [r331](../evidence/20261103-r337/campaign/patches/r331-vs-control309.patch) | taskqueue2 | — | — | — | rejected_framework_graph_compatibility |
| [r332](../evidence/20261103-r337/campaign/patches/r332-vs-r319.patch) | paged-suffix | 750.1/0.3578/1.3328 | 1149.7/0.5312/1.6295 | 1751.1/0.8127/2.3549 | no_consistent_gain_or_inconclusive |
| [r333](../evidence/20261103-r337/campaign/patches/r333-vs-control309.patch) | taskqueue0 | 908.3/0.4068/1.1664 | 1345.2/0.6223/1.4621 | 2155.9/1.0019/1.8947 | no_consistent_gain_or_inconclusive |
| [r334](../evidence/20261103-r337/campaign/patches/r334-vs-control309.patch) | low-pressure-split | 774.5/0.3508/1.3496 | 1193.2/0.5300/1.7418 | 1615.6/0.7946/2.4186 | promising_needs_reverse_confirmation |
| [r335](../evidence/20261103-r337/campaign/patches/r335-vs-control309.patch) | aclnn-cache100k | — | — | — | rejected_native_crash |
| [r336](../evidence/20261103-r337/campaign/patches/r336-vs-control309.patch) | pinned-pcm | 873.5/0.3760/1.2432 | 1240.8/0.5655/1.6328 | 1829.0/0.8691/2.2092 | no_consistent_gain_or_inconclusive |
| [r337](../evidence/20261103-r337/campaign/patches/r337-vs-r320.patch) | host-lengths | 759.1/0.3217/1.4676 | 1138.6/0.5168/1.7922 | 1626.7/0.7671/2.5127 | promising_needs_reverse_confirmation |
| [r338](../evidence/20261103-r337/campaign/patches/r338-vs-r332.patch) | mask-cache | 741.7/0.3499/1.3488 | 1164.2/0.5273/1.7347 | 1718.6/0.7957/2.4373 | no_consistent_gain_or_inconclusive |
| [r339](../evidence/20261103-r337/campaign/patches/r339-vs-control309.patch) | rope-invariant | — | — | — | rejected_diagnostic_crash |
| [r340](../evidence/20261103-r337/campaign/patches/r340-vs-control309.patch) | hift-exact-rows | 784.5/0.3601/1.3390 | 1293.0/0.6036/1.5701 | 1692.3/0.7873/2.4339 | no_consistent_gain_or_inconclusive |
| [r341](../evidence/20261103-r337/campaign/patches/r341-vs-control309.patch) | script-elementwise | 787.1/0.3735/1.3078 | 1257.5/0.5517/1.6757 | 1810.7/0.8452/2.2724 | no_consistent_gain_or_inconclusive |
| [r342](../evidence/20261103-r337/campaign/patches/r342-vs-r320.patch) | low-pressure-split | 737.6/0.3242/1.4396 | 1226.5/0.5556/1.6452 | 1601.0/0.7751/2.4713 | no_consistent_gain_or_inconclusive |
| [r343](../evidence/20261103-r337/campaign/patches/r343-vs-control309.patch) | audio-stream-cores | 835.2/0.3356/1.3947 | 1203.4/0.5237/1.7759 | 1629.4/0.7725/2.4889 | promising_needs_reverse_confirmation |
| [r344](../evidence/20261103-r337/campaign/patches/r344-vs-control309.patch) | euler-exact-rows | 816.1/0.3876/1.2159 | 1263.9/0.5795/1.6017 | 1770.3/0.8092/2.3558 | no_consistent_gain_or_inconclusive |
| [r345](../evidence/20261103-r337/campaign/patches/r345-vs-control309.patch) | rounding-native-rms | 804.8/0.3615/1.3168 | 1206.3/0.5515/1.6558 | 1731.5/0.8327/2.3138 | no_consistent_gain_or_inconclusive |
| [r346](../evidence/20261103-r337/campaign/patches/r346-vs-control309.patch) | native-rotary | 788.4/0.3615/1.3357 | 1149.2/0.5381/1.7051 | 1830.5/0.8642/2.2196 | no_consistent_gain_or_inconclusive |
| [r347](../evidence/20261103-r337/campaign/patches/r347-vs-r337.patch) | mask-cache | 772.4/0.3290/1.4575 | 1166.6/0.5198/1.7572 | 1719.0/0.7840/2.4343 | no_consistent_gain_or_inconclusive |
| [r348](../evidence/20261103-r337/campaign/patches/r348-vs-control309.patch) | thinker-static-fia | 872.1/0.3825/1.2648 | 1227.6/0.5557/1.6530 | 1893.9/0.8755/2.1847 | no_consistent_gain_or_inconclusive |
| [r349](../evidence/20261103-r337/campaign/patches/r349-vs-r337.patch) | cores-third | 791.8/0.3342/1.4208 | 1228.5/0.5306/1.7696 | 1783.3/0.8392/2.2601 | no_consistent_gain_or_inconclusive |
| [r350](../evidence/20261103-r337/campaign/patches/r350-vs-r346.patch) | rounded-native-rotary | 776.2/0.3618/1.3057 | 1184.4/0.5401/1.6991 | 1769.8/0.8159/2.3750 | no_consistent_gain_or_inconclusive |
| [r351](../evidence/20261103-r337/campaign/patches/r351-vs-r343.patch) | host-lengths | 792.6/0.3443/1.3772 | 1193.1/0.5336/1.7370 | 1718.3/0.8107/2.3162 | no_consistent_gain_or_inconclusive |

[最终确认、完整失败日志和机制证据](../evidence/20261103-r337/README.md)。
