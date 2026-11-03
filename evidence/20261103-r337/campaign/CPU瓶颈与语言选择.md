# CPU瓶颈与语言选择

当前结论：没有观察到持续GIL饱和，也没有证据支持优先把HTTP前端整体改写为C++/Rust。**尚不能排除主机提交端的局部瓶颈。** 主机调用、CANN workspace/任务更新、设备等待和纯Python计算必须分开。

CPU/GIL证据主要来自当前冻结工作负载的C8窗口，不能直接外推为C16/C32或其他请求分布也没有CPU瓶颈。

## 已有有效证据

| 观测 | 结果 | 能说明什么 |
| --- | --- | --- |
| 外部`/proc` CPU时钟，19.91秒、133个观测点 | 引擎82.47%单核，滚动1秒最高91.55%；音频35.76%；HTTP2.26%；服务约2.95核 | 引擎主机线程值得继续研究；不代表纯Python占用，也不能按整机CPU低而排除单线程限制 |
| 外部py-spy，只采GIL持有者，20秒、100Hz | 名义2000个机会中620条持有者样本，约31%；引擎330、音频187、HTTP50 | 该诊断窗口未见持续GIL饱和；样本包含持有GIL的C扩展，不能当作纯Python时间占比 |
| 恢复的CPU→queue→CANN→设备trace | 104,950对enqueue/dequeue；排队中位2.61µs、P95 167.8µs | 有短任务和长尾；平均值及host span不能直接当成CPU计算耗时 |
| 设备事件窗口 | 7,636.636ms窗口，计算/拷贝并集5,867.640ms，未覆盖1,768.996ms | 仍有空隙；没有证明这些空隙全由CPU、Python或GIL造成 |
| 局部C++执行实验r341 | 12种独立算子case、四次重复均精确；16条Thinker检查通过；C2/C4/C8未取得一致收益 | 仅减少Python表达式派发并不保证端到端改善；不能据此证明所有C++方案无用 |

`/proc`和GIL数据来自不同诊断窗口，不能将82.47%减31%解释为CANN时间。音频同步调用包含等待设备执行，不是纯CPU工作。设备busy比率也不是算力利用率/MFU。

## 拒绝使用的CPU归因

- 自定义`thread_time`的cProfile出现负数和超出窗口的函数耗时，已判为`INVALID_TIMER_ATTRIBUTION`。
- 原生`perf_event_open`返回`EPERM`，未修改容器权限或系统安全设置。
- Austin帧指针采样遗漏仍有CPU活动的engine/audio线程调用栈，不能使用其表面上的ORT占比。
- AustinP/libunwind在服务中自身SIGSEGV，只有四字节头，无有效栈。

这些失败均保留原始产物与退出码，不用于计算“Python占比”或性能收益。

## 补充：容器限额和调度计数

在r351独立C8诊断进程上，20.035秒外部`schedstat`运行时增量约为：引擎73.85%单核、音频26.26%、准备23.09%、HTTP主线程2.02%。这仍是包含原生调用的线程CPU时间，且来自带诊断的进程，不用于性能验收。

该机器使用cgroup v1；最初读取的v2路径不存在，未将缺失值当作“无限额”。按实际挂载读取v1后，容器`cpu.cfs_quota_us=-1`、周期100000µs，cpuset与有效CPU均为`0-191`；`nr_throttled=0`、`throttled_time=0`。未观察到本容器自身CFS配额限流，不能据此排除宿主机其他负载/祖先cgroup或单线程发射能力限制。

`kernel.sched_schedstats=0`，因此线程runqueue等待数据明确置为不可用，没有把零值解释成“没有CPU竞争”。未修改任何配额、亲和性或内核选项。

证据：[独立诊断调度分析](scheduler-cpu-analysis.json)、[v1容器配额](cgroup-v1-quota-snapshot.json)。

## 继续采用的判断顺序

1. 从设备空隙反查提交lead、队列与具体Python/CANN路径，确认哪一步阻止后续任务入队；仅有host长span不足以证明因果。
2. 优先复用可信的主机元数据、减少逐层D2H观察和小算子调用、明确共享scratch所有权。用独立原生benchmark验证净效果。
3. 对可隔离热点尝试C++执行、原生融合或图化，保留精度边界与fallback；不能把“用了C++”当作收益原因。
4. 只有当HTTP解析、编码、音频CPU处理或GIL竞争在关键路径上被证实，才扩大到C++/Rust前端或多进程重构。新增序列化、IPC、复制及NPU context开销也要计入。

多进程共享卡、同进程多stream、stream核配额是不同机制。r320/r343测试的是本进程/音频stream的Cube/Vector核限制，没有部署或证明NVIDIA MPS式守护进程机制。r343同时显式改变了图捕获stream，已安排保留捕获方式、关闭quota的对照，避免把两项影响混为一谈。

Omni新鲜基线的实际资源审计记录到同一张NPU上三个stage模型进程；独立引擎使用单服务进程内的engine/audio/prepare线程和stream。这是已核对的结构差异，但尚未测出跨进程context切换或IPC各占多少时间，不能把总体差值全部归因于它。两者都运行Python，语言名称本身不能解释性能。

## 尚存的主机依赖

源码仍有Talker块结束后一次token D2H、逐请求EOS/chunk处理，以及PCM传回主机后发送。这些既可能包含不可避免的数据依赖，也可能存在可重排的等待；搜索到`.cpu()`不等于可以直接删除。图捕获预热中的`synchronize()`也不能当成每个稳态请求的重复开销。

因此，后续C++/Rust方案应针对被证实阻塞提交的具体小段，与原路径做隔离对照。单纯把Python循环换语言而保留相同同步、队列依赖和CANN调用，未必改善TTFP或RTF。

## 验收

以官方C2/32、C4/64、C8/128请求的TTFP mean、RTF mean、request/s作为性能指标；采样器开启时的计时不参与验收。保留前后对照和反向顺序重复，输出时长/批次轨迹变化也需保留。新音频候选仍必须通过冻结speech gate，当前评分环境不可达，因此尚无新的等质性能发布或默认版本提升。

证据：[CPU时钟](native-cpu-analysis.json)、[GIL](gil-probe-analysis.json)、[队列桥接](profilee-bridge-analysis.json)、[原生采样失败审计](native-profiler-validation.json)、[完整逐轮实验](持续优化实验记录.md)。
