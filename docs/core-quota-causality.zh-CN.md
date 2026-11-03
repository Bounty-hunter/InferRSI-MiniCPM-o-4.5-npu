# 单进程少用 NPU 核为何更快的固定工作量验证

## 本次结论

2026 年 11 月 3 日，完成固定工作量单组件对照和核配额 × 串行/多 stream 的 2×2 实验。固定输入下，12 Cube/24 Vector 配额让 Talker 和 Token2Wav 执行更快，让 Thinker 执行更慢。串行任务包也有收益，因此配额收益包含独立于跨组件并发的执行路径改善。两个配额下，多 stream 均缩短了任务包时间，较低配额下的绝对节省没有扩大。[结果汇总](../evidence/20261103-core-quota/summary.json)

这里的 12/24 指 NPU Cube/Vector 资源配额。单推理进程仍可提交多条设备 stream，并使用后端运行时线程。核配额会影响设备算子的分块与执行路径，本次配对 profile 已观察到相同 shape 的 `Block Num` 和耗时变化。具体 CANN 内部 tiling 决策尚未完全证明。[配对 kernel 分析](../evidence/20261103-core-quota/profile-kernel-analysis.json)

**跨配额输出发生可复现的数值变化，最终 speech gate 为 `NOT_RUN`。** 此次证据支持执行时间与数值路径判断，质量验收仍待完成。[源码与验收状态](../evidence/20261103-core-quota/source-manifest.json)

## 1. 实验控制

| 项目 | 固定内容 |
|---|---|
| 硬件 | Ascend910B1，物理 NPU 2 |
| 容器 | `dyy-v032-r182`，镜像 `9fd63cd20600`，UID/GID `1033:1000` |
| 调度 | 原 `run_with_npu_device_group.sh`，容器内 `npu run` |
| 源码 | r320 runtime，tree SHA256 `842af75958017cd767a5cd8841f162ded0aa59085c3f86db0dc52308d670b44a` |
| 模型 revision | `503e754207c94da6bb26850b4469f367c9ea3582` |
| 计算配置 | Flow FP16、6 步、chunk25、counter RNG，双方相同 |
| 唯一配置变量 | `npu_core_fraction`，A 为 1.0，B 为 0.5 |
| 实际配额 | A 为 24 Cube/48 Vector，B 为 12 Cube/24 Vector |
| 进程序列 | A1 → B1 → B2 → A2 → A3 → B3，6 个新推理进程 |
| 输入来源 | 16 条原 Seed-TTS 请求的诊断捕获，冻结真实组件输入、prefix/KV 和音频 cache |

每个单组件预热 5 次，再测量 20 次。下表先取每个进程的中位数，再对同配额的 3 个进程求算术平均。计时边界为调用前设备同步完成，到调用后设备同步完成，包含该调用的 host 提交和完成等待。另保留 stream event 时间。样本量支持工程对照，未计算最终统计门的置信下界。[逐对分析](../evidence/20261103-core-quota/fixed-work-analysis.json) · [执行脚本](../evidence/20261103-core-quota/scripts/replay_fixed.py)

## 2. 固定工作量单组件对照

| 组件与固定任务 | A，24/48 ms | B，12/24 ms | B 相对 A |
|---|---:|---:|---:|
| Thinker，batch8 一次 decode，每请求 1 token，hidden `[8,4096]` | 25.6715 | 28.8475 | 慢 12.37% |
| Talker，batch4 的 K10 块 | 33.0139 | 30.5757 | 快 7.39% |
| Token2Wav，batch4，每项固定 28-token 音频窗口及 cache | 147.7618 | 123.4063 | 快 16.48% |

三个组件均直接重放冻结输入，调用次数、batch 和窗口保持一致。固定 workload 消除了完整生成时 token 数、终止长度和调度批次改变的混杂。每个配额内重复输出逐位一致，6 个进程使用同一组输入文件 SHA256，输入模板检查均通过。[输入 manifest](../evidence/20261103-core-quota/capture/manifest.json) · [6 个进程原始样本](../evidence/20261103-core-quota/trials/)

Token2Wav 在这组固定任务中占时较大，它的改善超过了 Thinker 回退。该权重取决于实际 workload，无法推广成所有 batch、shape 都应降低配额。

## 3. 核配额 × 串行/多 stream 的 2×2

一个固定任务包包含一次 Thinker、一次 Talker、一次音频窗口。Thinker 与 Talker 在 engine stream 顺序提交，音频在另一条 stream 提交。串行模式在提交音频前等待 engine stream，多 stream 模式允许两条 stream 重叠，最终均等待设备完成。**所有任务由一个 host 线程提交。** 每进程每种模式计时 12 次，交替切换模式顺序，再按上述中位数与三进程均值聚合。[任务包实现](../evidence/20261103-core-quota/scripts/replay_fixed.py)

| 核配额 | 串行 ms | 多 stream ms | 多 stream 绝对节省 ms |
|---|---:|---:|---:|
| A，24/48 | 204.2742 | 176.9656 | 27.3086 |
| B，12/24 | 180.5556 | 154.6850 | 25.8706 |
| B 相对 A 时延降低 | 11.61% | 12.59% | — |

串行下的 11.61% 改善说明，减少组件间竞争无法单独解释全部收益。多 stream 的绝对节省由 27.31 ms 变为 25.87 ms，这组固定包没有观察到降低配额扩大绝对重叠收益。两个配额都从多 stream 获益。[2×2 原始汇总](../evidence/20261103-core-quota/summary.json)

另一组任务包将 engine 工作增加到四次 Thinker+Talker，音频仍为一次固定窗口。

| 核配额 | 串行 ms | 多 stream ms |
|---|---:|---:|
| A，24/48 | 374.2703 | 346.5540 |
| B，12/24 | 351.3139 | 325.3755 |
| B 相对 A 时延降低 | 6.13% | 6.11% |

改变组件工作占比后，配额收益缩小。这与 Thinker 回退、音频改善的方向一致。真实服务由不同 host 线程驱动组件，并涉及动态排队和请求依赖，本次固定重放无法拆出服务流水线各机制的完整贡献比例。这些毫秒值的计时范围是固定任务完成时间，服务级 TTFP、RTF、request/s 继续沿用原生 benchmark 的测量口径。

## 4. 相同 shape 的设备算子发生了什么

A/B 各采集一次配对诊断 profile，均包含 3 个串行和 3 个多 stream 的四次 engine 任务包。CPU/NPU activity 开启，`record_shapes=True`、`with_stack=False`、`PipeUtilization`。双方各 266538 条 kernel 记录，652 个算子/shape 组全部匹配且调用次数一致。以下为带 profiler 的设备平均耗时，只用于定位路径变化。[kernel 分析与原始 CSV 路径/SHA256](../evidence/20261103-core-quota/profile-kernel-analysis.json)

| 同 shape 算子 | A → B 平均耗时 µs | A → B 的 Block Num |
|---|---:|---:|
| Talker FusedInferAttentionScore，query `[4,12,64]` | 28.290 → 23.384 | 24 → 12 |
| Flow LayerNorm，`[8,50,512]` | 20.121 → 12.316 | 45 → 24 |
| Cast，`[8,50,512]` | 9.212 → 4.365 | 48 → 24 |
| Thinker MatMul，`[8,4096] × [24576,4096]` | 153.154 → 230.266 | 24 → 12 |
| HiFT ConvTranspose，output `[4,8,1,6000,8]` | 4056.613 → 155.813 | 32 → 12 |

最后一行在每个配额中各出现 6 次，其中位数为 4058.48 → 155.98 µs。对应 `aic_scalar_ratio` 约 0.926 → 0.661，`aic_mac_ratio` 约 0.025 → 0.346。这是特定 shape 的执行路径差异，整体 Token2Wav 收益以上文未开启 profiler 的计时为准。

`Block Num` 表示启动分块数，物理核占用还需结合执行映射判断。相同 shape 与调用次数，加上分块及计数变化，支持核配额改变算子分块/执行路径的解释。精确到 CANN 内部 tiling、workspace、带宽和跨 stream 竞争各占多少，目前证据仍有限。此轮未采 Python stack，因此也未对 CPU 发射瓶颈作因果归因。跨 stream kernel 时间可能重叠，kernel 耗时之和保持诊断口径。

## 5. 数值检查与质量边界

| 固定组件输出，B 相对 A 捕获值 | 检查结果 |
|---|---|
| Thinker hidden | max_abs 0.375，relative L2 约 0.014254 |
| Talker codec | token 序列发生变化 |
| 音频 | 波形发生变化，比较向量均长 96000 samples，relative L2 约 1.3340 |

A 重放输出与捕获值逐位一致。B 在三个新进程中产生一致的另一组输出，每个进程内的重复检查也通过。双方模型、精度、步数和输入保持相同，但改变核配额后，底层数值路径出现可复现变化。codec 是离散编号，对编号计算相对 L2 缺少语音质量解释。音频波形 L2 同样不能代替既定声学质量评分。最终 speech gate 仍为 `NOT_RUN`，本次保留为配额因果诊断证据。[数值与时延汇总](../evidence/20261103-core-quota/summary.json)

## 6. 审计、复核与执行状态

6 个计时进程、2 个 profile 进程及预约外层回执均退出 0。计算已结束，本次预约已释放，收尾检查无本任务排队项。初始捕获遇到失效的默认模型缓存路径，改为显式模型目录后成功。首轮元数据采集遇到包装 API 无法使用 `inspect.getsource`，改为后端源文件 fingerprint 后成功。用户明确范围前启动的额外服务复测已取消，本报告全部数值来自固定工作量实验。

紧凑证据包含输入/源码哈希、真实配额观察、原始计时样本、退出码、输出摘要和执行脚本。大输入、完整日志与原始 profile 保留在 NPU 主机。仓库校验器复算紧凑证据与绑定关系，大文件内容的再次核验需要访问远端。

```powershell
python tools/verify_core_quota_study.py
```

远端宿主机作业日志位于 `/home/dengyunyang/omni-npu-jobs/minicpmo-core-components-r4-20261103/`。容器产物根目录为 `/workspace/minicpmo-r182/artifacts/core-causality-20261103/`，对应宿主机 `/home/dengyunyang/omni-npu-work/minicpmo-v032-main-r182-20261026/artifacts/core-causality-20261103/`。大输入位于其 `fixed-inputs-r2/`，profile CSV 的精确路径与 SHA256 保存在 kernel 分析内。[证据索引](../evidence/20261103-core-quota/README.md)
