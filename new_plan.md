<!-- Copyright (C) 2026 Intel Corporation -->

# upstream openvino.genai 多级 KV Cache Offloading 架构设计与实施计划（含可插拔 SSD Plugin 扩展）

> **本文档是最终权威方案。** 后续所有施工、评审、验收均以本文档（`C:\Users\gta\openvino.genai\new_plan.md`）为准。
> 仓库里另一份 `kv_cache_offload_new_plan.md` 记录的是按更保守范围（无租户按请求隔离、无 SSD 插件 C ABI、无 GPU
> host↔device 批量 swap kernel）实施并已验证的 Phase 0-6 工作；它与本文档的设计有实质分歧（尤其是租户隔离的
> 挂载点、`IKVCacheStorageBackend` 的接口形状），**以本文档为准，`kv_cache_offload_new_plan.md` 只作为
> "哪些机制性代码已经存在、可以复用"的现状参考，不再作为独立的权威计划。**
>
> **文档定位**：本文档为将单进程 KV Cache Offloading 方案从 `pipeline.mx` 移植并升级至 upstream `openvino.genai`（`genai.mx`）与 `openvino` runtime（`openvino.mx`）的权威架构设计规范与落地工程计划。
> **核心目标**：在 upstream `openvino.genai` 的 Continuous Batching 框架中构建原生的三级缓存体系（L0 Device → L1 Host DRAM → L2 Local Storage/SSD），实现跨 Session 极速前缀复用、断电/重启持久化缓存、强租户隔离，并提供 **Vendor SSD Plugin 扩展架构**，支持第三方 SSD 硬件厂商（如 Solidigm、Samsung、Kioxia、Intel 等）接入定制化驱动（SPDK、io_uring、ZNS/FDP、GPUDirect/USM Direct I/O）以最大化硬件吞吐。
> **基线代码**：`thirdparty/openvino.genai`（ov_master）与 `thirdparty/openvino`（ov_master）。

---

## 0. 现状核对与重新施工计划（2026-09-07）

在本文档被确认为权威方案之前，已经按 `kv_cache_offload_new_plan.md` 的（更窄）范围完成并验证了一部分工作
（对应 git 历史 `Phase 0`–`Phase 6 (partial)` 共 6 次提交）。本节如实盘点这部分工作与本文档设计的对应关系，
分三类：**可直接复用**、**需要按本文档重塑**、**全新工作（当前 0% 完成）**，并给出接下来的施工顺序。

### 0.1 可直接复用（机制已验证，接口/命名可能需要小改）

| 已完成的机制 | 现有代码位置 | 对应本文档章节 |
| --- | --- | --- |
| Store 触发（LRU block 覆写前保存）、host 快照同步、磁盘异步写 | `continuous_batching/cache/kv_cache_offload_cache.{hpp,cpp}` | §3.2（L0→L1 搬运的一半：淘汰触发） |
| L1 独立 host 内存池（LRU、容量上限、与磁盘写解耦） | `continuous_batching/cache/kv_cache_host_block_pool.hpp` | §3.2.1 `HostBlockPool` |
| 固定 slot 磁盘文件、slot 分配/回收、checksum、跨重启 manifest 持久化、崩溃安全写入顺序 | `continuous_batching/cache/kv_cache_offload_manager.{hpp,cpp}` | §3.3.3 `DefaultFileStorageBackend`（需要重新实现批量 I/O + metadata 结构，见 0.2） |
| Prefix 恢复计划（`warm_prefix_cache`、批量 `load_into_many`、首次失败截断语义） | `continuous_batching/cache/block_manager.hpp` | §4.2 Restore 时序（无 L2 SSD 插件命中分支，需要补） |
| 设备侧真实批量传输（`RemoteTensor` ROI 拷贝，block-major→segment-major 重排） | `continuous_batching/cache/kv_cache_manager.hpp`（`write_blocks`/`write_segment_run`） | §3.2.2 阶段 1（USM/RemoteTensor），**不是**阶段 2 的专用 Swap Kernel（见 0.3） |
| Python `SchedulerConfig`/`CacheOffloadConfig` 绑定模式（field-by-field，避免位置参数错位） | `src/python/py_continuous_batching_pipeline.cpp` | §5.1（字段列表需要按本文档改，绑定写法可复用） |

### 0.2 需要按本文档重塑（现有实现的设计取舍和本文档不一致）

1. **租户隔离挂载点错了，需要重做**：现有实现把 `tenant_id`/`cache_salt` 挂在 `SchedulerConfig`（每个
   pipeline 实例一个 tenant），在 `CacheOrchestrator` 构造 `BlockManager` 时算一次根种子传进去。本文档
   §3.1.2 要求挂在 `GenerationConfig`（**按请求级别**），直接在 `Sequence::_make_hash` 里于 block-0 注入
   虚拟根种子。这是两种不同粒度的设计，必须重做：
   - 需要在 `SequenceGroup`/`GenerationConfig` 上新增 `tenant_id`/`cache_salt` 字段和访问器。
   - 需要改 `sequence_group.cpp` 里的 `Sequence::_make_hash`，在 `filled_blocks_count == 0` 分支注入种子
     （而不是像现在这样在 `BlockManager::seeded_hash()` 里统一包一层）。
   - **注意本文档 §3.1.2 自身有一处矛盾必须先澄清再实现**：示例代码在两者皆为空时不注入种子（等价于
     "不隔离"，向后兼容），但紧接的文字又说默认应生成"单次会话级 Ephemeral 随机 Salt"（等价于默认禁止
     跨 session 复用，与本文档 §1.1 第①条目标冲突）。**采纳代码语义**（空则不注入，显式设置才隔离），
     并在实现时把这条默认策略在代码注释和本文档里都写清楚，不能再自相矛盾。
   - 现有的 `SchedulerConfig::tenant_id`/`cache_salt`/`compute_prefix_isolation_seed()`/
     `BlockManager::seeded_hash()` 在改造后需要整体移除，避免两套隔离机制并存造成混淆。
2. **`IKVCacheStorageBackend` 接口形状需要重新定义**：现有接口（`i_kv_cache_storage_backend.hpp`）是
   "同步、单 slot、无 metadata、无 batch、无 async"的最小接口。本文档 §3.3.1 要求的接口包含
   `StorageModelMetadata`（模型/tokenizer 指纹 + KV 精度/布局）、`BlockIORequest`（按 hash 而非 slot 索引寻址，
   显式 `host_buffer`/`layer_idx`）、批量 `read_blocks`/`write_blocks`、`async_read_blocks` 预取接口、
   `has_blocks` 批量查询。需要重新设计接口并迁移 `KVCacheOffloadManager` 的实现去匹配，而不是照抄现有接口。
   `KVCacheOffloadCache` 内部"按 hash 查 slot、LRU 淘汰"这部分逻辑可以保留，只是它和 backend 交互的方式要
   从"slot 索引"改造成"hash + 显式 buffer 描述符"。
3. **`CacheOffloadConfig`/`SchedulerConfig` 字段命名要对齐本文档 §5.1**：现有字段名
   （`cache_offload_config.capacity_bytes`/`enable_persistence` 等）需要映射或改名为
   `host_cache_size`、`storage_cache_dir`、`storage_cache_size`、`storage_backend_type`、
   `storage_plugin_path`、`storage_plugin_properties`。
4. **manifest/持久化的 hash 稳定性问题需要保留现有做法、不能用 `std::hash`**：本文档 §3.1.2/§3.3.3 的示例
   代码用 `std::hash<std::string_view>`，但 `std::hash` 的具体实现不保证跨进程/跨构建稳定，不能用来做
   跨重启持久化 manifest 的 key。**这一点采纳现有实现的做法**（`kv_cache_offload_manager.cpp` 里手写的
   FNV-1a），本文档里凡是"落盘 key"相关的 hash 计算，实现时都改用 FNV-1a 而不是 `std::hash`；`std::hash`
   仅用于内存态、进程内生命周期的哈希（现状已经是如此，不用改）。

### 0.3 全新工作，目前 0% 完成（2026-09-08 更新：澄清真实诉求）

**真实诉求澄清**：本节最初把 GPU 侧的缺口理解为"host↔device 批量拷贝不够快"，这是错的。跟用户当面对齐
后确认，真实诉求是 **GPU 端 KV cache 目前只能是一整块连续显存：扩容必须整体重新分配 + 全量拷贝旧数据，
且从来没有"归还显存"这个概念（block 被逻辑淘汰后，物理显存永远留在那一整块 tensor 里）**。用户需要的是
**可收缩、可真正归还给驱动**的非连续显存管理；`std::vector<size_t> src_block_ids, dst_block_ids` 批量
搬移 kernel 的真正用途是**碎片整理**——把还在用的 block 从要清空的 chunk 搬到其它 chunk 腾出空间，腾空后
才能把整块 chunk 销毁、显存真正还给驱动，而不是 host↔device 拷贝优化。

**已核实的现状**：
- `KVCacheManager::allocate_cache_if_needed()`（`continuous_batching/cache/kv_cache_manager.hpp`）目前
  每层只维护**一个** `ov::Tensor`（经 `m_context.create_tensor()` 分配）；扩容时分配一块更大的连续
  tensor，把旧数据整个 `copy_from` 过去——没有"新增一块独立、不连续显存"的概念，也没有释放机制。
- `paged_attention` primitive（`src/plugins/intel_gpu/include/intel_gpu/primitives/paged_attention.hpp`）
  的 `KEY_CACHE`/`VALUE_CACHE` 输入（`PagedAttentionInputIdx::KEY_CACHE = 3`）是**单个**、编译期定形的
  tensor，同一 batch 内所有 sequence 的 block 都假定落在这一块连续显存里——这是"计算热区本身要支持不
  连续 chunk"时必须要改的地方。
- `PA_KV_Reorder`（同目录 `pa_kv_reorder.hpp`）做的是**单一连续 tensor 内部**的 gather/scatter，服务于
  cache eviction 重排/压缩，**不能直接复用**，只在"block 索引类输入怎么传给 kernel"这件事上有参考价值
  （`CreatePA_KV_ReorderOp`/`PaKVReorderFusion` 证实它也是编译进 `ov::Model` 图里的算子，不是可以在
  model graph 之外被直接调用的工具函数）。
- GPU plugin 里 `allocation_type::usm_device` 是成熟的一等公民（`ocl_memory.cpp`/`ze_memory.cpp`/
  `sycl_memory.cpp` 均支持），可以作为"USM 设备指针能被当数值存进另一块 buffer、供 kernel 内部解引用"
  这个技术点的分配基础；但**没有在这个代码库里找到"kernel 参数是指针数组、kernel 内部解引用"这个具体
  模式的现成先例**，是本 phase 里唯一需要单独实验验证、不能直接大规模改造后才发现跑不通的技术不确定性。

**设计方案（已与用户确认选择方案 A：计算热区本身支持不连续 chunk，而非另建一层不可算的溢出池）**：
- 固定每个 chunk 的 block 容量为 N，令 `block_id = chunk_id * N + local_offset`，chunk 定位只需一次
  整数除法/取模，现有 `BLOCK_INDICES` 机制不用改。
- `paged_attention` 新增一个输入 `CHUNK_BASE_PTRS`：一个固定上限大小（如 64）的 USM 设备指针数组，未用
  槽位填 null；kernel 内部地址计算从 `base_ptr + block_id * stride` 改为
  `chunk_base_ptrs[block_id / N] + (block_id % N) * stride`。要求每个 chunk 必须是 USM device 分配
  （而不是插件默认可能选择的普通 `cl_mem` buffer）。
- 新增一个跨 chunk 批量搬移 primitive（碎片整理用），结构上参考 `pa_kv_reorder`，但寻址逻辑要用同一套
  chunk 索引方案，因为 src/dst block 可能落在不同 chunk。

**子阶段划分（Phase 5 内部）**：
- **5a（技术验证）—— 已验证通过**：写了一个独立、可抛弃的最小实验（不依赖 `openvino`/`openvino.genai`
  构建系统，动态加载 `OpenCL.dll` 并通过 `clGetExtensionFunctionAddressForPlatform` 解析
  `cl_intel_unified_shared_memory` 扩展函数），在真实硬件（Intel(R) Arc(TM) 140T GPU）上验证：两块独立
  `clDeviceMemAllocINTEL` 分配（模拟两个不连续 chunk），把各自的设备指针存进一个 `clHostMemAllocINTEL`
  分配的指针表里，单次 kernel launch 里按 `chunk_id = block_id / blocks_per_chunk` 查表解引用、读取
  正确 chunk 的数据——8/8 个 block 全部读取正确，2000 次迭代平均单次 launch 2.39 us（无异常开销）。
  **这一步的技术不确定性已经排除，可以继续 5b。**
- **5b（`paged_attention` 改造）—— 已完成**：新增 `CHUNK_BASE_PTRS` 输入（primitive 输入个数
  28→29，`has_chunk_base_ptrs = inputs.size() == 29`）；`BLOCKS_PER_CHUNK` 是 JIT 编译期常量，
  `NUM_CHUNKS` 是运行时 scalar kernel 参数（在每次 dispatch 时从 `chunk_base_ptrs` 的实际 runtime
  shape 读取，而非烘焙进 kernel JIT cache key，避免 chunk 数增长时反复重新编译 kernel）；写 kernel
  （`pa_kv_cache_update_ref.cl`）与读 kernel（`paged_attention_opt.cl`）的 decode/prefill 分支均已
  接入 chunk 寻址（`chunk_id = block_id / BLOCKS_PER_CHUNK`，`local_block_idx = block_id % BLOCKS_PER_CHUNK`）。
  **真实 bug 与修复**：USM 指针是被 kernel 间接解引用的（从 `chunk_base_ptrs` 参数 buffer 里读出来
  再当指针用），OpenCL 要求这种间接访问必须显式通过
  `clSetKernelExecInfo(CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL, ...)` 告知驱动，否则驱动不建立页表映射，
  执行时以 `CL_OUT_OF_RESOURCES` 报错（且异常经常发生在未被 catch 的线程上，直接 `std::terminate`，
  无可读错误信息）。已在 `ocl_stream.cpp::set_arguments()` 里修复：按 `PagedAttentionInputIdx::CHUNK_BASE_PTRS`
  的声明下标精确匹配（不能用"最后一个 INPUT 参数"这种位置猜测，否则会在非 chunked 的 kernel 分派上
  误触发，已实测导致 xattention 回归崩溃，随后修正为按下标匹配）。已验证：GPU 单测（paged_attention +
  xattention）340 项全部通过，真实 Intel Arc GPU 硬件。
- **5c（跨 chunk 批量搬移 primitive）—— 未开始**：碎片整理用的新 primitive，`src_block_ids`/`dst_block_ids`
  批量搬移，一次 Kernel Launch 完成，寻址复用 5b 的 chunk 索引方案。**这是当前唯一还没做的部分**：
  没有这个 primitive，chunk 只能增长、不能真正压缩/归还给驱动（逻辑上淘汰的 block 腾出的空间无法让
  整个 chunk 变空，也就无法销毁该 chunk）。
- **5d（`openvino.genai` 侧改造）—— 已完成**：`KVCacheManager` 改为"每层一个 chunk 列表"（`m_key_chunks`/
  `m_value_chunks`，`std::vector<std::vector<ov::Tensor>>`），`allocate_chunked_cache_if_needed()`
  只增量追加新 chunk（USM device 分配，`shared_mem_type = USM_DEVICE_BUFFER`，否则拿到的是普通
  `cl_mem`，指针不可解引用——这是另一个已修复的真实 bug），从不重新分配/拷贝已有 chunk；每次分配后
  调用 `refresh_chunk_base_ptrs_tensor()` 刷新 `chunk_base_ptrs.N` 并 `set_tensor` 绑定。
  `SchedulerConfig` 新增 `use_chunked_kv_cache`/`kv_cache_chunk_size_blocks`，与
  `use_cache_eviction`/`enable_prefix_caching`/`enable_kv_cache_offloading` 互斥（构造期硬断言）。
  尚未做的是**收缩触发逻辑**（何时发起碎片整理、何时真正销毁空 chunk）——这依赖 5c 先落地。
- **5e（端到端验证）—— 已完成**：真实 Qwen3-8B、Intel Arc GPU 上端到端生成，chunked 与非 chunked
  的生成文本逐字一致（`TestChunkedKVCache.ProducesTheSameTextAsWithoutChunking`，
  `tests/cpp/chunked_kv_cache_e2e.cpp`）；`openvino.genai` 侧回归 163 项测试 158 通过 + 5 跳过（无关
  的 offload 测试）；`openvino` 侧 GPU 单测 340 项全部通过。
  **性能验证（用 `tools/llm_bench` 对真实模型做的，非模拟）**：稳态场景（`num_kv_blocks` 固定为一个
  够用的值，全程不触发扩容）下，chunked 比不 chunked 慢约 11-12%（TTFT 84.5ms→97.0ms，
  吞吐 13.68→12.16 tokens/s）——这是间接寻址（kernel 需要先从 `chunk_base_ptrs` 读指针再解引用）+
  每次 dispatch 都要 `clSetKernelExecInfo` 登记的固定开销，从第一个 token 起就存在，与是否真的发生过
  扩容无关。这笔"分块税"是恒定的；"扩容不用整体拷贝"这部分收益要等 5c（收缩）落地后才能完整评估
  （目前只能证明"不倒退"，没有可对比的"收缩省下多少"）。
  **验证过程中发现并修复了一个真实 bug**：`num_kv_blocks=0`/`cache_size=0`（动态显存模式）下开启
  chunked KV cache 会在每次 `generate()` 结束时崩溃（access violation）——根因是 `generate()` 末尾
  "动态模式下 batch 结束就清空 cache" 的优化（`ContinuousBatchingPipeline::generate()` 里的
  `clear_cache()` 调用）对 chunked cache 不安全：`KVCacheManager::clear()` 直接销毁了 chunk 的 USM
  设备内存，却没有同步解绑/更新已经绑定在 infer request 上的 `chunk_base_ptrs.N` 输入，留下悬空指针。
  这条代码路径此前从未被任何固定 `num_kv_blocks` 的测试覆盖过（包括本文档的 chunked E2E 测试）。
  修复：`clear_cache()` 的触发条件里加上 `!use_chunked_kv_cache`，chunked cache 干脆不参与这个"空闲
  归还"优化（与 5c 尚未实现"收缩"的现状一致）。顺带修复了 `Logger::write_message()` 从不 flush 的
  bug（导致排查这次崩溃时一开始看不到任何 `GENAI_INFO`/`[KV_TRACE]` 输出）。修复后：崩溃复现脚本
  （`max_new_tokens` 分别取 5/10/30，均在此前必现崩溃）全部通过；`llm_bench` 动态模式跑 1024 token
  生成正常完成；`openvino.genai` 回归 163 项测试、`TestChunkedKVCache.*` E2E 测试均重新验证通过，
  无回归。详见 repo memory `openvino-gpu-paged-attention-kv-cache.md` 第 8c 节。
  **尚未做**：长上下文/高并发场景下用系统工具观察显存占用曲线、吞吐/TTFT 对比——这类"收缩后确实
  归还显存"的收益验证依赖 5c 先落地（当前只能验证"增长不重新拷贝"这部分收益，还没有"收缩"可言）。


### 0.4 修订后的施工顺序

在原第 6 节路线图的基础上，按"先改隔离机制 → 再重塑存储接口 → 再做 GPU 新 kernel → 最后端到端评测"重新排序，
详见下方第 6 节已更新的表格（Phase 1 现在明确标注为"重做"，Phase 2/3 标注为"复用+改造"，Phase 4/5 标注为
"新建"并给出真实文件落点）。

---

## 1. 架构目标与设计原则

### 1.1 核心诉求与业务场景
1. **单 Session 内与跨 Session Prefix Caching**：长系统提示词（System Prompt）、多轮对话（Multi-turn Chat）、Few-shot 模板等前缀 KV Block 零重复计算。
2. **跨进程 / 实例重启持久化**：服务重启或模型重新加载后，历史高频 Prefix KV Block 从本地 SSD 毫秒级恢复，消除冷启动延迟。
3. **超长上下文（Long Context）显存卸载**：当 GPU/NPU 显存吃紧时，自动将非活跃、远端或即将被淘汰的 KV Block 下沉至 Host 内存或 SSD，避免 OOM 并大幅提升并发 Capacity。
4. **强租户与会话安全隔离**：原生内置 `tenant_id` / `cache_salt` 链式哈希污染机制，防止跨租户越权命中与侧信道信息泄漏。
5. **硬件厂商生态开放（SSD Plugin）**：提供轻量、高性能、解耦的存储抽象接口与 C/C++ ABI 动态载入机制，允许 SSD 厂商提供专属优化的硬件插件，替代默认 POSIX 文件后端。

### 1.2 架构设计原则（Karpathy Guidelines & OpenVINO Core）
- **In-Tree 原生契合**：深度复用 `openvino.genai` 已有的 `Sequence::_make_hash`、`OverwritableBlocksHashStore`、`CacheOrchestrator` 等基础设施，不破坏现有连续批处理调度模型。
- **正交分层与极简契约**：L0/L1/L2 各层只对抽象 Block 及其 Hash 负责，存储后端与设备计算流水线解耦。
- **Surgical Changes**：对核心调度主循环改动保持最小化，仅在 Block 分配（Allocate）、淘汰（Evict）、恢复（Restore）三个生命周期节点插入分层感知逻辑。
- **渐进增强（Graceful Degradation）**：若未配置 SSD 路径或插件加载失败，系统平滑退化为 L0 单层或 L0+L1 双层模式，保证零崩溃。

---

## 2. 总体分层架构图

```
+----------------------------------------------------------------------------------------------------+
|                                    User Application / Facade                                       |
|  GenerationConfig(tenant_id, cache_salt)  |  SchedulerConfig(enable_kv_cache_offloading, ssd_...)    |
+----------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+----------------------------------------------------------------------------------------------------+
|                                  openvino.genai Continuous Batching                                 |
|                                                                                                    |
|  +-----------------------------------------------------------------------------------------------+  |
|  |                                      Scheduler & Pipeline                                     |  |
|  |   - SequenceGroup / Sequence (tenant_id 混入虚拟 Block-0 链式哈希)                                |  |
|  |   - PrefixRestorePlan (生成包含 L0 / L1 / L2 来源的多级恢复计划)                                 |  |
|  +-----------------------------------------------------------------------------------------------+  |
|                                                |                                                   |
|                                                v                                                   |
|  +-----------------------------------------------------------------------------------------------+  |
|  |                                       CacheOrchestrator                                       |  |
|  |  +----------------------------+  +----------------------------------------------------------+ |  |
|  |  |   BlockManager (L0 元数据)   |  |            TieredCacheManager (分级缓存协调器)            | |  |
|  |  | - m_prefix_hash_to_blocks  |  | - L0/L1/L2 块迁移状态追踪                                  | |  |
|  |  | - OverwritableBlocksStore  |  | - 异步 Swap-Out / Swap-In 队列与流控                       | |  |
|  |  | - LRU / Eviction Policy    |  | - 多级淘汰与命中判定策略 (Hit/Miss Decision Engine)         | |  |
|  |  +----------------------------+  +----------------------------------------------------------+ |  |
|  +-----------------------------------------------------------------------------------------------+  |
+----------------------------------------------------------------------------------------------------+
          |                                      |                                    |
          v                                      v                                    v
+-----------------------+              +-------------------+               +-------------------------+
|   L0: Device Cache    |  D2H Copy    |  L1: Host Cache   |  Storage IO   |    L2: Storage Cache    |
| (GPU VRAM / NPU USM)  | <=========>  | (Pinned Host RAM) | <===========> |    (Local NVMe / SSD)   |
|                       |  H2D Copy    |                   |               |                         |
| KVCacheManager        |              | HostBlockPool     |               | IKVCacheStorageBackend  |
| (OpenVINO Tensor/USM) |              | (Aligned Memory)  |               | (Plugin Architecture)   |
+-----------------------+              +-------------------+               +-------------------------+
                                                                                        |
                                                          +-----------------------------+-----------------------------+
                                                          |                                                           |
                                                          v                                                           v
                                            +---------------------------+                               +---------------------------+
                                            |   DefaultFileStorage      |                               |   Vendor SSD Plugin       |
                                            | (In-Tree Default Backend) |                               | (Solidigm/Samsung/Custom) |
                                            |                           |                               |                           |
                                            | - POSIX pread/pwrite/mmap |                               | - SPDK Polling Driver     |
                                            | - Container Block File    |                               | - Linux io_uring / Direct |
                                            | - JSON/Binary Manifest    |                               | - ZNS / FDP Flash Stream  |
                                            | - Zero-dependency         |                               | - USM/GPUDirect P2P DMA   |
                                            +---------------------------+                               +---------------------------+
```

---

## 3. 核心子系统详细设计

### 3.1 安全隔离与链式哈希扩展（Tenant Isolation & Hash Scheme）

#### 3.1.1 漏洞根因与防线突破
标准 vLLM 的 `cache_salt` 仅混入第 0 个物理块的哈希计算中；如果匹配请求从第 1 个块之后切入，或者攻击者构造了相同后缀，存在越权利用公共前缀的风险。

#### 3.1.2 链式传播哈希设计
在 `openvino.genai` 中，`Sequence::_make_hash`（`src/cpp/src/sequence_group.cpp`）基于前一个块的哈希值与当前块 Token 序列迭代计算。
我们将 `tenant_id` 与可选 `cache_salt` 封装为 **Sequence 根种子（Virtual Parent Hash）**：

```cpp
// Sequence Group / Sequence 内部哈希计算更新
size_t Sequence::_make_hash(size_t content_length, size_t block_size) {
    auto sequence_group = get_sequence_group_ptr();
    size_t block_start_idx = content_length - (content_length % block_size);
    if (block_start_idx == content_length) {
        block_start_idx -= block_size;
    }

    std::vector<int64_t> content;
    size_t filled_blocks_count = block_start_idx / block_size;

    if (filled_blocks_count > 0) {
        // 继承前驱块哈希
        content.emplace_back(m_prefix_hashes[filled_blocks_count - 1]);
    } else {
        // Block-0: 注入租户与 Salt 虚拟根种子
        const std::string& salt = sequence_group->get_cache_salt();
        const std::string& tenant_id = sequence_group->get_tenant_id();
        if (!salt.empty() || !tenant_id.empty()) {
            std::string root_seed = tenant_id + ":" + salt;
            size_t seed_hash = std::hash<std::string_view>{}(root_seed);
            content.emplace_back(static_cast<int64_t>(seed_hash));
        }
    }

    // 压入当前块的 Tokens / Embeddings ...
    // [原有 Token/Embedding 填充逻辑保持不变]

    const char* data = reinterpret_cast<const char*>(content.data());
    std::size_t size = content.size() * sizeof(content[0]);
    return std::hash<std::string_view>{}(std::string_view(data, size));
}
```

- **数学性质保证**：由于哈希链结构，只要 Block-0 的种子不同，后续所有衍生 Block 的哈希值均严格隔离，数学上彻底根除跨租户碰撞。
- **安全默认行为**：若未显式指定 `tenant_id`，系统默认生成单次会话级 Ephemeral 随机 Salt，**仅允许当前 Session 内复用，禁止跨 Session/磁盘复用**。

---

### 3.2 L1 Host Cache 与 L0↔L1 搬运机制

#### 3.2.1 HostBlockPool（主机内存池）
- 预分配锁页内存（Pinned Host Memory），避免 OS 换页引入抖动。
- 维护固定大小对齐内存块（按 `block_size_in_bytes` 划分），由 `HostBlockAllocator` 统一管理。
- 维护 `m_l1_hash_to_block` 哈希索引与 LRU 淘汰链表。

#### 3.2.2 跨设备搬运（D2H / H2D）
针对不同推理硬件，在 `KVCacheManager` 中实现两阶段搬运能力：
1. **CPU 推理后端**：
   - GPU/NPU 显存与 Host 内存为同一物理介质。
   - `swap_out` / `swap_in` 退化为 `std::memcpy` 或指针所有权置换，吞吐可达 40~60 GB/s。
2. **GPU 推理后端（阶段 1 - 标准 OpenVINO USM / RemoteTensor）**：
   - 利用 `ov::RemoteContext` 与 `ov::RemoteTensor` 的 ROI 切片拷贝：
   - Host 侧使用 USM Host 内存，通过 Level Zero / OpenCL 异步 DMA 队列进行传输。
3. **GPU 推理后端（阶段 2 - 显存可收缩/可归还的分 chunk 管理，非 host↔device 拷贝）**：
   - 在 `openvino` GPU plugin 中让 `paged_attention` 的计算热区支持读取多个不连续的 device chunk（详见
     §0.3 的完整设计与 5a-5e 子阶段划分），并新增一个跨 chunk 批量搬移 primitive：支持传入
     `std::vector<size_t> src_block_ids, dst_block_ids`，一个 Kernel Launch 完成数十个分散 Block 的并发
     搬移，用于碎片整理——把还在用的 block 集中到少数 chunk，腾空的 chunk 才能整体销毁、真正归还显存。

---

### 3.3 可插拔 SSD 插件架构（Extensible SSD Plugin Architecture）

为支持 Solidigm、Samsung、Intel 及其他存储硬件厂商提供极致优化的 SSD 驱动实现，采用 **C++ 抽象接口 + 动态 C ABI 插件载入机制**。

```
                    +------------------------------------------+
                    |          IKVCacheStorageBackend          |
                    |           (C++ Core Interface)           |
                    +------------------------------------------+
                                         ^
                                         | C++ Adapter Wrapper
                    +------------------------------------------+
                    |          ov_genai_ssd_plugin_t           |
                    |          (C ABI Plugin Interface)        |
                    +------------------------------------------+
                                         ^
                                         | dlopen / LoadLibrary
                    +--------------------+---------------------+
                    |                                          |
        +-----------------------+                  +-----------------------+
        | libov_genai_ssd_samsung.so |             | libov_genai_ssd_solidigm.so |
        | (e.g. io_uring / FDP) |                  | (e.g. SPDK / CSAL / ZNS)    |
        +-----------------------+                  +-----------------------+
```

#### 3.3.1 C++ 抽象契约（`IKVCacheStorageBackend`）

```cpp
namespace ov::genai {

struct StorageModelMetadata {
    std::string model_name;
    std::string model_fingerprint; // 结构/权重 Hash
    std::string tokenizer_hash;
    ov::element::Type kv_precision;
    size_t kv_block_size_tokens;
    size_t num_layers;
    size_t block_bytes_per_layer;
};

struct BlockIORequest {
    uint64_t block_hash;
    void* host_buffer;         // 对齐的 Host 内存指针（支持 USM/Pinned）
    size_t buffer_size_bytes;
    uint32_t layer_idx;
};

using AsyncIOCompletionCallback = std::function<void(int status_code, size_t completed_blocks)>;

class IKVCacheStorageBackend {
public:
    virtual ~IKVCacheStorageBackend() = default;

    /// 初始化插件环境、挂载存储路径、校验/重建元数据
    virtual void initialize(const std::string& storage_path, 
                            const StorageModelMetadata& metadata,
                            const ov::AnyMap& custom_properties) = 0;

    /// 检查指定 Hash 的 Block 在存储介质中是否存在
    virtual bool has_block(uint64_t block_hash) const = 0;

    /// 批量查询哪些 Hash 存在于存储中
    virtual std::vector<bool> has_blocks(const std::vector<uint64_t>& block_hashes) const = 0;

    /// 同步写入 Block 数据（包含所有 Layer）
    virtual bool write_blocks(const std::vector<BlockIORequest>& requests) = 0;

    /// 同步读取 Block 数据（包含所有 Layer）
    virtual bool read_blocks(const std::vector<BlockIORequest>& requests) = 0;

    /// 异步批量读取（供 Prefetch 流水线使用）
    virtual void async_read_blocks(const std::vector<BlockIORequest>& requests,
                                   AsyncIOCompletionCallback callback) {
        // 默认实现退化为同步调用
        bool ok = read_blocks(requests);
        callback(ok ? 0 : -1, ok ? requests.size() : 0);
    }

    /// 淘汰指定 Block 或根据存储预算执行清理
    virtual void evict_blocks(const std::vector<uint64_t>& block_hashes) = 0;

    /// 存储元数据落盘与同步
    virtual void flush_manifest() = 0;

    /// 释放存储资源
    virtual void shutdown() = 0;
};

} // namespace ov::genai
```

#### 3.3.2 C ABI 导出规范（供第三方 vendor 编写动态库）

第三方厂商只需编译独立的 `.so` / `.dll`，无需依赖完整的 OpenVINO 构建树：

```c
// include/openvino/genai/c/ssd_plugin_interface.h
#ifndef OPENVINO_GENAI_SSD_PLUGIN_INTERFACE_H
#define OPENVINO_GENAI_SSD_PLUGIN_INTERFACE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* model_name;
    const char* model_fingerprint;
    const char* tokenizer_hash;
    uint32_t element_type;
    size_t kv_block_size;
    size_t num_layers;
    size_t block_bytes_per_layer;
} ov_genai_storage_metadata_t;

typedef struct {
    uint64_t block_hash;
    void* buffer;
    size_t buffer_bytes;
    uint32_t layer_idx;
} ov_genai_block_io_desc_t;

typedef struct ov_genai_ssd_plugin {
    uint32_t api_version; // 校验 API 兼容版本，例如 0x00010000
    const char* vendor_name; // 如 "Solidigm-SPDK", "Samsung-FDP"
    
    void* (*create_instance)(void);
    void (*destroy_instance)(void* handle);
    
    int (*init)(void* handle, const char* path, const ov_genai_storage_metadata_t* meta, const char* json_props);
    int (*has_block)(void* handle, uint64_t hash);
    int (*read_blocks)(void* handle, const ov_genai_block_io_desc_t* descs, size_t count);
    int (*write_blocks)(void* handle, const ov_genai_block_io_desc_t* descs, size_t count);
    int (*evict_blocks)(void* handle, const uint64_t* hashes, size_t count);
    int (*flush)(void* handle);
} ov_genai_ssd_plugin_t;

// 动态库必须导出的入口符号
ov_genai_ssd_plugin_t* ov_genai_get_ssd_plugin(void);

#ifdef __cplusplus
}
#endif

#endif // OPENVINO_GENAI_SSD_PLUGIN_INTERFACE_H
```

#### 3.3.3 DefaultFileStorageBackend（内置默认实现）
- **存储文件结构**：采用 `kv_cache_data.bin` 紧凑连续大文件（按 4KB 对齐） + `manifest.json` 元数据索引。
- **I/O 引擎**：Linux 环境优先使用 `pread`/`pwrite` 或 `io_uring`，Windows 环境使用 `ReadFile`/`WriteFile` 重叠 I/O。
- **元数据安全性**：每次持久化写入追加 CRC32 校验码，启动时校验模型签名，若不一致则自动清空重建，杜绝脏数据注入。

#### 3.3.4 厂商特定扩展点与高级加速特性
1. **Solidigm / Intel CSAL & SPDK 插件**：利用用户态 NVMe 轮询驱动绕过 Linux Kernel VFS 与中断，4K/16K Block 读延迟从 15μs 降至 3~5μs。
2. **Samsung FDP (Flexible Data Placement) / ZNS 插件**：将不同 Tenant / 模型的 KV Block 分配到不同 Flash Stream / Physical Zone，彻底消除 SSD 内部垃圾回收（GC）引起的尾延迟（Tail Latency）。
3. **GPUDirect Storage (GDS) / USM Direct DMA**：厂商插件可直接通过 PCIe Peer-to-Peer DMA 从 NVMe 读写 GPU USM / Pinned 内存，完全绕过 CPU Host 中转。

---

## 4. 多级生命周期调度与决策时序

### 4.1 Block 淘汰时序（Eviction / Swap-Out）

```
[Scheduler Step Finished]
         |
         v
[BlockAllocator::allocate_block] --> L0 物理块已满?
         |
        YES
         v
[OverwritableBlocksHashStore::get_lru_block_to_overwrite]
         |
         v
[TieredCacheManager::notify_l0_eviction(hash, blocks)]
         |
         +---> 1. 触发 D2H 拷贝将 Block 搬运至 L1 HostBlockPool
         |
         +---> 2. 若 L1 Host 内存亦满，挑选 L1 LRU Block 触发 Storage 写入：
                   IKVCacheStorageBackend::write_blocks(...)
         |
         v
[L0 物理槽位被安全覆写给新 Sequence]
```

### 4.2 Block 命中与多级恢复时序（Restore / Swap-In）

```
[New Request Arrived: Prompt Tokens]
         |
         v
[SequenceGroup::compute_prefix_hashes(tenant_id, salt)]
         |
         v
[BlockManager::get_prefix_restore_plan]
         |
         +--- Tier 0 (Device) 命中? ----> [YES] --> 标记 L0 复用 (Plan.l0_blocks)
         |
         +--- Tier 1 (Host) 命中? ------> [YES] --> 标记 L1 恢复 (Plan.l1_blocks)，分配 L0 块并在 prefill 前下发 H2D Swap-In
         |
         +--- Tier 2 (SSD Plugin) 命中? -> [YES] --> 标记 L2 恢复 (Plan.l2_blocks)，分配 L0/L1 块并在 prefill 前下发 SSD->Host->Device Swap-In
         |
         v
[执行增量 Prefill (仅对未命中的剩余 Tokens 计算)]
```

---

## 5. 公共 API 与配置定义

### 5.1 `SchedulerConfig` 扩展（C++ & Python）

```cpp
struct SchedulerConfig {
    // ... [原有参数保持] ...

    // 是否开启多级 KV Cache Offloading
    bool enable_kv_cache_offloading = false;

    // L1 Host RAM 缓存大小（单位：GB，0 表示不使用 L1 Host 缓存）
    size_t host_cache_size = 0;

    // L2 存储路径（若为空则不开启 L2 存储缓存）
    std::string storage_cache_dir = "";

    // L2 存储大小上限（单位：GB）
    size_t storage_cache_size = 0;

    // 存储后端类型："default" 或 "plugin"
    std::string storage_backend_type = "default";

    // 动态载入的第三方 SSD 插件路径（例如 "libov_genai_ssd_solidigm.so"）
    std::string storage_plugin_path = "";

    // 传递给 SSD 插件的厂商专有参数
    ov::AnyMap storage_plugin_properties = {};
};
```

### 5.2 `GenerationConfig` 扩展

```cpp
struct GenerationConfig {
    // ... [原有采样参数保持] ...

    // 租户隔离标识符（强制作用于 Block 链式哈希根种子）
    std::string tenant_id = "";

    // 用户自定义 Cache Salt 种子
    std::string cache_salt = "";
};
```

---

## 6. 实施计划与里程碑（Implementation Roadmap）

> 状态列如实反映 2026-09-07 的现状（见第 0 章）：**复用**＝已有代码可以直接搬过来用；**改造**＝已有机制存在，
> 但接口形状/挂载点要按本文档重做；**新建**＝目前 0% 完成，需要从头设计实现。

| 阶段 | 状态 | 交付模块 | 涉及源码路径 | 验收标准 / 测试用例 | 依赖项 |
|---|---|---|---|---|---|
| **Phase 1** | **已完成** | **安全隔离与链式根哈希改造**：把 `tenant_id`/`cache_salt` 从 `SchedulerConfig`（pipeline 级）迁移到 `GenerationConfig`（请求级），直接改 `Sequence::_make_hash` 在 block-0 注入根种子；移除了原有 `compute_prefix_isolation_seed()`/`BlockManager::seeded_hash()` 这一套 pipeline 级实现（`BlockManager` 构造函数、10 处调用点全部恢复为无 seed 版本）。`KVCacheOffloadManager` 的 `tenant_isolation_seed`（manifest 持久化隔离）暂时固定传 0，留给 Phase 3 存储接口重塑时解决。 | `src/cpp/src/sequence_group.cpp`<br>`src/cpp/include/openvino/genai/generation_config.hpp` | 编写单元测试：验证不同 `tenant_id`/`salt` 下相同 Prompt 生成完全不同的 Prefix Hash 链，相同 `tenant_id` 生成一致 Hash。（补充：空 `tenant_id`+空 `cache_salt` 时哈希须与未隔离版本逐位相同，向后兼容；见 §0.2 第 1 条对本文档 §3.1.2 默认策略矛盾的澄清）——**已验证**：`TestBlockManager.DefaultTenantConfigMatchesUnseededSequenceHash`/`DifferentTenantIdsProduceDifferentHashesForIdenticalContent`/`SameTenantIdProducesSameHashForIdenticalContent` 三个新测试通过；完整 716 项测试套件（17 项已知无关失败）与真实 GPU E2E 均无回归。 | 无 |
| **Phase 2** | **已完成** | **L1 Host Memory Pool 与 L0↔L1 搬运**：`kv_cache_host_block_pool.hpp` 的独立 LRU host 池（类名已经是 `HostBlockPool`，与本文档一致）和 `kv_cache_manager.hpp` 的 RemoteTensor 搬运已经实现并有真实 GPU 验证；`KVCacheOffloadCache`/`HostBlockPool` 补充了类级文档注释，明确对应本文档的 `TieredCacheManager`/`HostBlockAllocator` 概念（`HostBlockAllocator` 未拆分为独立类：当前规模下由 `HostBlockPool` 自身的 LRU 簿记直接承担，避免为了对齐命名而引入不必要的间接层；锁页/预分配内存也是已记录的未实现项，留给真正的 DMA 瓶颈出现时再做）。 | `src/cpp/src/continuous_batching/cache/kv_cache_host_block_pool.hpp`<br>`src/cpp/src/continuous_batching/cache/kv_cache_manager.hpp`<br>`src/cpp/src/continuous_batching/cache/block_manager.hpp` | 单元测试：CPU/GPU 模拟高负载显存挤压，验证被淘汰 Block 成功迁入 L1 Host，二次请求命中后从 L1 恢复且生成 Token 完全一致。（现状：已有单元测试覆盖 CPU/GPU 搬运正确性，仍需补齐上述高负载挤压场景的端到端对比）——**已验证**：新增 `TestKVCacheOffloadCache.WarmPrefixCacheRestoresBlocksFromHostPoolUnderMemoryPressure` 单测（`BlockManager` 内存挤压 → L1 命中 → `num_load_disk==0` 确认走的是 L1 而非 L2）；真实模型 E2E（`kv_cache_offload_e2e.cpp`）新增 `host_cache_slots=16` 配置，`ProducesTheSameTextAsWithoutOffload`/`DoesNotGrowDeviceMemory` 在真实 Intel Arc GPU 上验证 L1 命中时生成文本与不开启 offload 时逐字一致；完整 717 项测试套件（17 项已知无关失败）无回归。 | Phase 1 |
| **Phase 3** | **已完成** | **L2 存储抽象接口与 DefaultFileStorage**：按 §3.3.1 重新定义 `IKVCacheStorageBackend`（`StorageModelMetadata`/`BlockIORequest`/批量 `read_blocks`/`write_blocks`/`async_read_blocks`/`has_blocks`），把现有 `kv_cache_offload_manager.{hpp,cpp}` 的 slot 分配、checksum、FNV-1a manifest、崩溃安全写入顺序迁移过去实现新接口，而不是重新发明这些机制。（现状：接口实际落地在 `src/cpp/src/continuous_batching/cache/i_kv_cache_storage_backend.hpp`/`kv_cache_offload_manager.{hpp,cpp}`，未新建 `include/openvino/genai/cache/i_storage_backend.hpp`/`default_file_storage.cpp` 两个路径——沿用仓库既有的 `continuous_batching/cache/` 目录结构，避免无谓的文件搬迁；`BlockIORequest` 按用户明确要求使用逐层（`layer_idx`）粒度而非整块粒度；`CacheOffloadConfig` 按 §5.1 完成字段改名/新增：`path`→`storage_cache_dir`，新增 `storage_cache_size`/`storage_backend_type`/`storage_plugin_path`/`storage_plugin_properties`（后三者为 Phase 4 预留，误用即在 pipeline 构造期抛异常，不做静默忽略）；`KVCacheOffloadManager` 构造函数移除 `device` 参数，设备支持性检查改为调用方（`CacheOrchestrator::enable_kv_cache_offload`）显式调用 `is_supported_device()` 后再构造） | `src/cpp/src/continuous_batching/cache/i_kv_cache_storage_backend.hpp`<br>`src/cpp/src/continuous_batching/cache/kv_cache_offload_manager.hpp`<br>`src/cpp/src/continuous_batching/cache/kv_cache_offload_manager.cpp`<br>`src/cpp/src/continuous_batching/cache/kv_cache_offload_cache.{hpp,cpp}`<br>`src/cpp/include/openvino/genai/cache_offload.hpp`<br>`src/cpp/src/continuous_batching/cache/cache_orchestrator.hpp`<br>`src/python/py_continuous_batching_pipeline.cpp` | 单测与集成测试：进程退出后重启 Pipeline，验证从 `manifest.json` 与 `.bin` 成功加载历史前缀，TTFT 降低 80% 以上。（补充：已有的 checksum 损坏/manifest 记录撕裂/model-tokenizer-layout 指纹不匹配安全重建等测试需要在新接口下原样保留）——**已验证**：`tests/cpp/kv_cache_offload_manager.cpp`（15 个用例，改写为 hash 寻址+逐层请求）与 `tests/cpp/kv_cache_offload_cache.cpp`（33 个用例，含第三方 mock backend `InMemoryMockStorageBackend` 验证接口边界、以及跨重启持久化恢复用例）全部改写通过，checksum 损坏由旧接口的抛异常改为新接口约定的布尔返回值（`read_blocks`/`write_blocks` 返回 `false`），已在测试中同步验证；完整 718 项测试套件（17 项已知无关失败 + 1 项已确认与本次改动无关的 `AUTO` 插件间歇性 flake，单独重跑 5 次全部通过）无回归；真实 Intel Arc GPU E2E（`TestKVCacheOffloadEndToEnd.*`）4/4 通过。 | Phase 2 |
| **Phase 4** | **已完成** | **SSD Plugin 动态加载架构与 C ABI 封装**：新增 `ov_genai_ssd_plugin_t` C ABI（`api_version` 校验、`create_instance`/`init`/`has_block(s)`/`read_blocks`/`write_blocks`/`evict_blocks`/`flush_manifest`/`shutdown`/`get_num_slots`/`get_num_free_slots`/`get_recovered_hashes`/`describe`）与对应的 `KVCacheOffloadSSDPluginBackend`（`IKVCacheStorageBackend` 适配器，负责 `dlopen`/`LoadLibrary`、符号解析、API 版本校验、`init()` 失败处理，全部失败路径直接抛异常，不做静默回退）；`CacheOrchestrator::enable_kv_cache_offload` 按 `storage_backend_type`（`"default"`/`"plugin"`）在两个 backend 之间显式分派。（现状说明：C ABI 相比本文档 §3.3.2 示例做了两处刻意的、已文档化的简化——`properties_string` 用 `key=value;...` 扁平字符串代替 JSON blob，避免任何一侧引入 JSON 依赖；`get_recovered_hashes` 用"先查数量、再填充"的两段式调用代替返回堆分配数组，避免跨 ABI 边界的分配器不匹配问题；插件 backend 要求所有 decoder layer 的 key+value 字节数一致，这是 `ov_genai_storage_metadata_t::block_bytes_per_layer` 为单一标量、无法表达非均匀 layer 尺寸的直接后果，构造期以 `OPENVINO_ASSERT` 显式拒绝不满足条件的模型，而不是静默截断/出错。） | `src/cpp/include/openvino/genai/c/ssd_plugin_interface.h`<br>`src/cpp/src/continuous_batching/cache/kv_cache_offload_ssd_plugin_backend.{hpp,cpp}`<br>`src/cpp/src/continuous_batching/cache/cache_orchestrator.hpp`<br>`src/cpp/include/openvino/genai/cache_offload.hpp`<br>`tests/cpp/mock_ssd_plugin/`（`mock_ssd_plugin.cpp`/`not_a_plugin.cpp` 及各自 CMake 目标） | 编写 Mock Vendor Plugin（`mock_ssd_plugin.so`），测试动态发现、载入、读写委托与异常故障隔离测试。（补充：插件显式配置但加载/运行失败时必须报错，不允许静默回退到 default backend，见 §0.2 对 §7 风险应对里"静默回退"提法的修正）——**已验证**：新增 3 个测试用 shared library 目标（`mock_ssd_plugin`——正常实现，支持通过 `properties_string` 注入 `fail_init`/`fail_write`/`fail_read` 故障；`mock_ssd_plugin_bad_version`——`api_version` 故意不匹配；`not_a_plugin`——真实可加载但不导出所需符号的库），均刻意不链接 OpenVINO/openvino.genai，证明 C ABI 边界确实零依赖；`tests/cpp/kv_cache_offload_ssd_plugin_backend.cpp` 12 个用例覆盖文件不存在/符号缺失/版本不匹配/`init()` 失败均抛异常、正常读写驱逐委托往返、批量 `has_blocks`、`write_blocks`/`read_blocks` 失败按接口约定返回 `false` 而非抛异常、非均匀 layer 尺寸提前拒绝；完整 730 项测试套件（17 项已知无关失败 + 最多 3 项已确认与本次改动无关的 `AUTO` 插件间歇性 flake——`TestKVCacheOffloadCache.*`/`TestScheduler.*` 中随机一两个测试，单独重跑 5 次全部通过）无回归；真实 Intel Arc GPU E2E（`TestKVCacheOffloadEndToEnd.*`）4/4 通过，证明默认 backend 路径未受影响。 | Phase 3 |
| **Phase 5** | **新建（5a 已验证；5b 进行中——读写两条 kernel 路径的非压缩 cache 分支均已完成，`openvino` 全量编译通过，且新增的 chunk 寻址逻辑本身已在真实 GPU 单元测试中直接验证通过——见下方"5b 单元测试验证"）** | **GPU 端可收缩/可归还的非连续显存管理（不是 host↔device 拷贝优化）**：需要在 `openvino`（不是 `openvino.genai`）的 Intel GPU plugin 里让 `paged_attention` 的计算热区支持读取多个不连续的 device chunk，并新增一个跨 chunk 碎片整理 primitive，详细设计与 5a-5e 子阶段划分见 §0.3。`PA_KV_Reorder`（`src/plugins/intel_gpu/include/intel_gpu/primitives/pa_kv_reorder.hpp`）只在 block 索引/kernel 参数传递的写法上有参考价值，它本身是单一连续 tensor 内部的 gather/scatter，不能直接复用。<br><br>**5b 当前进度（均未编译，仅静态检查通过）**：① 核心 op `PagedAttentionExtension`/shape inference/template 参考后端已改为兼容 28 或 29 输入（新增可选输入 28 `chunk_base_ptrs`，向后兼容）；② GPU plugin 的 `cldnn::paged_attention` primitive 新增 `has_chunk_base_ptrs`/`blocks_per_chunk`/`num_chunks` 字段（含 hash/equality/序列化），`CreatePagedAttentionExtensionOp` 从 op 的 `rt_info["blocks_per_chunk"]` 与 `chunk_base_ptrs` 输入 shape 派生这些值；③ **写路径 kernel（`pa_kv_cache_update_ref.cl`/`KVCacheUpdateGenerator`）的非压缩（uncompressed）cache 分支——decode（2nd+ token）与 prefill（1st token，含整块/半块）均已实现 chunk 感知寻址**：物理 block id 通过 `chunk_id = id / BLOCKS_PER_CHUNK`、`local_block_idx = id % BLOCKS_PER_CHUNK` 解析，再从 `chunk_base_ptrs`（前 `NUM_CHUNKS` 个为 key chunk 指针、后 `NUM_CHUNKS` 个为 value chunk 指针）取出对应 chunk 的 USM 设备指针写入；`HAS_CHUNK_BASE_PTRS=0` 时 `local_block_idx` 直接等于原 `block_idx`，与改造前逐位一致。压缩/量化 cache（`IS_KV_COMPRESSED`，含 `IS_KEY_BY_CHANNEL`，因为该 flag 只在压缩模式下才会被置位）与 chunking 组合会触发编译期 `#error`，防止静默写错位置。**明确未完成、下一步必须做的**：主 attention 计算 kernel（`paged_attention_opt.cl`，读路径）的 decode（`pa_single_token`/`pa_gqa_single_token`）与 prefill-with-cache（`pa_multi_token`）分支——即真正读取 `key_cache`/`value_cache` 的唯一 kernel `KERNEL(pa_sdpa_opt)`——**现已完成非压缩 cache 分支的 chunk 感知寻址**（K 处理主循环 + V 处理主循环/尾块循环，均按 `chunk_id = block_indice / BLOCKS_PER_CHUNK`、`local_block_idx = block_indice % BLOCKS_PER_CHUNK` 解析后从 `chunk_base_ptrs` 取对应 chunk 指针；`HAS_CHUNK_BASE_PTRS=0` 时 `local_block_idx` 等于原值，逐位一致；压缩/量化分支组合同样触发编译期 `#error`）；对应的 `PagedAttentionGeneratorSingleToken`/`GeneratorMultiTokens`（`PagedAttentionGeneratorBase` 共享 `HAS_CHUNK_BASE_PTRS`/`BLOCKS_PER_CHUNK` jit 常量，各自 `get_arguments_desc()` 按位置插入 `CHUNK_BASE_PTRS` 参数）已同步更新（`NUM_CHUNKS` 后来发现不能是 jit 常量，见下方"5d 修正"）。写路径与读路径现已在 uncompressed cache 下"对称"完成，理论上具备端到端可用性。**已验证**：`cmake --build . --config RelWithDebInfo`（`C:\Users\gta\openvino\build`，设置 `HTTP_PROXY`/`HTTPS_PROXY=http://proxy-dmz.intel.com:912` 后）全量编译+链接通过（含 `openvino_intel_gpu_plugin.dll`、Python wheel 等全部 target，无 `error C`/`LNK`），日志见 `C:\Users\gta\openvino\build_log_phase5_retry.txt`。这只证明代码语法/类型/链接正确，**仍未做任何真实数据或 GPU 硬件端到端验证**（见下）。`pa_sdpa_opt`（C++ 类名 `PagedAttentionSDPAOptGeneratorMultiToken`，kernel 名 `"sdpa_opt"`）是完全不同的 kernel/文件，用于纯 prefill 自注意力（不读 cache），与 chunking 无关，未改动也不需要改动。`pa_kv_cache_rotate_ref.cl`（cache 淘汰重排）与整个 `cm/`（C-for-Media）后端仍完全未动。**没有任何调用方会把 `has_chunk_base_ptrs` 置为 true**（`SDPAToPagedAttention` 的 opt-in 开关未实现，真实模型端到端仍无法触发 chunking）。<br><br>**5b 单元测试验证（已完成，2026-09-09/10）**：在 `paged_attention_gpu_test.h`/`.cpp` 里新增了直接构造 29-input（`has_chunk_base_ptrs=true`）primitive 的 GPU 单元测试路径——`paged_attention_test_params` 新增 `chunking_blocks_per_chunk` 字段（默认 0，纯新增、不影响任何既有用例的 positional brace-init）；`run_gpu_inference()` 在该字段 >0 时，把已经用真实数据填好的单一 `key_cache`/`value_cache` legacy buffer 按 `blocks_per_chunk` 物理块切成多个独立 `cldnn::memory`（默认 lockable USM 分配，`buffer_ptr()` 直接可用作 kernel 可见的原始设备指针），拼出 `chunk_base_ptrs` 表并设置 `has_chunk_base_ptrs`/`blocks_per_chunk`/`num_chunks`，`key_cache`/`value_cache` 两个 legacy 输入仍照常绑定（chunk 模式下是 kernel 内的死参数，避免改动 kernel 固定的参数列表）；由于 `PagedAttentionReference` 的基础比较逻辑完全来自 `PagedAttentionManager` 的 host 侧 query/key/value 向量、从不读取 GPU cache buffer 布局，同一套 CPU 参考对比对 chunked 和非 chunked 运行都成立，无需任何改动。新增 `smoke_paged_attention_chunked/paged_attention_chunked_test` 2 个用例：① decode 场景（`past_len=40`，`block_size=16`→3 个物理块，`blocks_per_chunk=2`→2 个 chunk，读 kernel 的逐块循环跨 chunk 边界，写 kernel 只碰 chunk1 的单个块）；② prefill 场景（`num_tokens=20`，`past_len=0`→2 个物理块：满块+4-token 半块，`blocks_per_chunk=1`→2 个 chunk，写 kernel 的整块/半块分支各自落在不同 chunk）。**两个用例均在真实 Intel Arc GPU 上跑通过**（`ov_gpu_unit_tests.exe --gtest_filter="*paged_attention_chunked_test*"`，2/2 PASSED），且完整现有 `paged_attention`/`xattention`/CM 相关测试套件（387 个用例，24 个既有 DISABLED）全部保持通过，证明本次 fixture 改动本身无回归。**这是本会话第一次真正跑通并验证 chunk 寻址逻辑本身**（而不仅仅是验证旧路径没被破坏）。仍未做的：`SDPAToPagedAttention` 的 opt-in 开关（真实模型端到端仍无法触发 chunking）、压缩/量化 cache 组合、cache-rotate kernel、CM 后端。<br><br>**5b 全量编译+回归验证（已完成，2026-09-08，早于上面的单元测试验证）**：`cmake --build . --config RelWithDebInfo`（`C:\Users\gta\openvino\build`）全量编译+链接通过（含 `openvino_intel_gpu_plugin.dll`、Python wheel，日志见 `build_log_phase5_retry.txt`）；用编译出的 wheel（`openvino-2026.4.0-22775-cp312-cp312-win_amd64.whl`）重装 `ov_export_venv`，配合该 venv 里已有的 `openvino_genai 2026.3.1.0`（跨版本引用，ABI 兼容，未重装/未重建 genai）在真实 Intel Arc GPU 上跑 `C:\Users\gta\e2e_tiny_llama`（随机权重的小模型，仅用于跑通管线，非真实语言质量验证）：① greedy 解码下 GPU 输出的最初若干 token 与 CPU 输出逐字一致，之后才因 CPU/GPU 浮点精度差异（该模型 logits 区分度极低，argmax 对微小误差极敏感）而分叉——这是预期内的正常现象，不是本次改动引入的 bug；② 同一 prompt 连续 3 次运行结果逐字确定性一致，未出现崩溃/挂起；③ 3 条 prompt 一起提交、由 continuous batching 并发调度生成 60 token，正常跑完无报错。**这只验证了 `HAS_CHUNK_BASE_PTRS=0`（现状默认、逐位向后兼容）这条路径在写 kernel 与读 kernel 上都没有回归**，该缺口已由上面的"5b 单元测试验证"补齐。<br><br>**5d 进行中（2026-09-10）**：开始把 5b 的能力接到 `openvino.genai` 真实调用链上。① `ov::pass::SDPAToPagedAttention`（`openvino/src/core/include/openvino/pass/sdpa_to_paged_attention.hpp`）新增第 8/9 个构造参数 `allow_chunked_kv_cache`/`kv_cache_blocks_per_chunk`（默认 false/512，向后兼容），`state_management_pattern.cpp` 在为 true 时按每层追加动态形状 `chunk_base_ptrs.N` Parameter（与既有 `key_cache.N` 走同一个 `pa_params.add()` 机制）并在 `PagedAttentionExtension` 节点 rt_info 写入 `blocks_per_chunk`；`openvino.genai` 侧新增 `SchedulerConfig::use_chunked_kv_cache`/`kv_cache_chunk_size_blocks`，在 `SchedulerConfig::validate()` 里显式声明与 `use_cache_eviction`/`enable_prefix_caching`/`enable_kv_cache_offloading` 三者互斥（构造期报错，不做静默降级），并透传到 `pipeline_impl.cpp` 的 `prepare_model_for_paged_attention()` 调用点。② `KVCacheManager`（`continuous_batching/cache/kv_cache_manager.hpp`）新增一条与既有单 tensor 路径完全隔离、通过 `m_chunked_kv_cache` 开关切换的 chunked 实现：`allocate_chunked_cache_if_needed()` 只追加全新 chunk（`m_context.create_tensor()`，USM 分配，永不重新分配/拷贝已有 chunk——这才是用户最初要的"扩容不再整体拷贝"）；`chunk_base_ptrs.N` 通过 `ov::intel_gpu::ocl::USMTensor::get()` 取每个 chunk 的原始设备指针后重建、`set_tensor` 刷新；`copy_blocks`（COW fork 必须支持，未与 eviction 互斥）新增 `copy_chunked_blocks()` 按 `(chunk_id, local_id)` 定位后在对应 chunk tensor 上做 ROI 拷贝；`get_key_cache`/`get_value_cache`（唯一单 tensor accessor，确认全仓库无任何调用方）、`clear()` 相应处理。构造调用点集中在 `cache_orchestrator.hpp::detect_cache_managers()` 一处。<br><br>**5d 修正（2026-09-10，编译前发现的设计缺陷）**：规划 `KVCacheManager` 时发现 5b 遗留问题——`chunk_base_ptrs` 的 shape 必须是动态的 `[-1]`（chunk 数量随生成不断增长），但 5b 把 `NUM_CHUNKS` 做成了 kernel 的**编译期 JIT 常量**，这站不住脚：静态 shape 断言会让模型编译失败，就算绕过去，chunk 数量一变就要重新 JIT 编译 kernel，代价不可接受。修正：保留 `blocks_per_chunk` 作为 jit 常量（配置常量，pipeline 生命周期不变），`num_chunks` 改为**运行时 scalar kernel 参数**——写 kernel（`KVCacheUpdateGenerator`）与读 kernel 的三个 generator（`PagedAttentionGeneratorSingleToken`/`GQASingleToken`/`MultiTokens`）都在 `get_dispatch_data_func()`（每次实际推理前调用，能拿到这一步 resolve 过的真实 runtime shape）里从 `params.input_layouts[CHUNK_BASE_PTRS]` 的实际 shape 现算 `num_chunks`，写进新增的 SCALAR 参数；`.cl` 侧 `chunk_base_ptrs[NUM_CHUNKS + chunk_id]` 相应改为 `chunk_base_ptrs[num_chunks + chunk_id]`（运行时变量）；`cldnn::paged_attention` primitive 的编译期 `num_chunks` 字段（含 hash/equality/序列化）整体移除，`CreatePagedAttentionExtensionOp` 不再断言 `chunk_base_ptrs` 是静态 shape。**已验证**：`ov_gpu_unit_tests`（`openvino/build`）全量重新编译通过；`smoke_paged_attention_chunked/paged_attention_chunked_test` 2/2 PASSED（确认运行时 scalar 改法功能不变）；完整 387 项 `paged_attention`/`xattention`/CM 回归套件全部 PASSED（24 项已知 DISABLED），无回归。仍未做的：`openvino.genai` 侧改动尚未编译验证（`kv_cache_manager.hpp`/`scheduler_config.hpp`/`pipeline_impl.cpp`/`cache_orchestrator.hpp` 的改动只做过语言服务器静态检查）；`BlockManager` 尚未验证"按整 chunk 增量取整"在真实调度循环里表现正确；尚无任何真实模型端到端测试。<br><br>**5d 编译验证（已完成，2026-09-14）**：把 `openvino.genai/build` 从默认链接的 pip 安装版 openvino（`C:/Python312/Lib/site-packages/openvino`，不含本会话任何改动）切换为直接指向本地开发包（`cmake -DOpenVINODeveloperPackage_DIR=C:\Users\gta\openvino\build`，需要先清空 `CMakeCache.txt`/`CMakeFiles` 消除旧 `TBB_DIR` 等缓存值残留造成的路径计算错误；`RelWithDebInfo` 配置要与 `openvino` 侧一致，否则链接期报 `LNK1181` 找不到 `openvino.lib`）。过程中发现并修正一处真实的构建依赖问题：`kv_cache_manager.hpp` 引入的 `openvino/runtime/intel_gpu/ocl/ocl.hpp`（为了用 `USMTensor::get()` 取原始设备指针）会连带要求 `CL/cl2.hpp`（OpenCL C++ 头，`openvino` 的第三方依赖，Developer Package 模式下不会暴露给下游），而 `openvino.genai` 的构建没有这个 include 路径；改为直接 `#include "openvino/runtime/intel_gpu/remote_properties.hpp"`（只是属性名+`void*`定义，无 OpenCL 依赖）并手写 `chunk_tensor.as<ov::RemoteTensor>().get_params().at(ov::intel_gpu::mem_handle.name()).as<gpu_handle_param>()`，绕开整个 `ocl.hpp`。**已验证**：`tests_continuous_batching.exe`（`openvino.genai/build`，`RelWithDebInfo`，链接本地含 chunk 改动的 `openvino`）全量编译通过；`--gtest_filter="*Scheduler*:*BlockManager*:*KVCache*"`（143 项，含 `mock_ssd_plugin`/`not_a_plugin` 动态库这次也一并构建齐全）138 PASSED + 5 SKIPPED（需要真实模型的 `TestKVCacheOffloadEndToEnd.*`，环境无模型正常跳过）+ 0 FAILED，证明本次改动在 `use_chunked_kv_cache=false`（默认）路径下对现有功能无回归。**这只验证了 openvino.genai 侧代码能正确编译、且不启用 chunking 时无回归**——`use_chunked_kv_cache=true` 这条路径本身当时仍然没有任何测试覆盖，已由下面"5d 单元测试验证"补齐。<br><br>**5d 单元测试验证（已完成，2026-09-14）**：新增 `tests/cpp/helper.{hpp,cpp}::get_dummy_model_with_chunk_base_ptrs()`（在既有 `get_dummy_model` 基础上，每层多加一个 `chunk_base_ptrs.N`（i64，动态 `[-1]`）`Parameter`，未被任何算子消费但仍会出现在 `compiled_model.inputs()` 里，足以让 `KVCacheManager::has_chunk_base_ptrs_inputs()` 检测通过——不需要真实的 `SDPAToPagedAttention`/`PagedAttentionExtension`，`KVCacheManager` 的分配/读写逻辑本身与真实 attention 计算完全解耦）。顺带补上一处防御性缺口：`read_block`/`write_block`/`write_blocks`（磁盘 offload 用的通用 block 级读写 API）之前完全没有处理 chunked 模式，会静默读写从未被真正写入数据的 legacy 哑元 tensor；现已对齐 `chunk_id_of`/`local_block_id_of` 寻址（`write_blocks` 的跨 block 连续 run 合并优化在 chunked 模式下退化为逐块调用，因为"连续"假设在跨 chunk 边界时不成立，且该批量路径本身只服务已与 chunking 互斥的磁盘 offload，退化足够安全）。新增 3 个 GPU-only 用例（`TestCacheManager.test_gpu_chunked_*`，无 GPU 时 `GTEST_SKIP`）：① 分配按 `blocks_per_chunk` 整数倍取整（6 块请求→分配 2 chunk=8 块，非精确 6 块）；② **增长后已写入的旧 chunk 数据保持不变**（写 block 1 → 从 1 chunk 增长到 3 chunk → 读回 block 1 逐字节相同）——这是 chunking 相对于"整体拷贝重分配"的核心价值主张，第一次被直接验证；③ `copy_blocks`（COW fork）跨 chunk 边界正确定位。**三个用例均在真实 Intel Arc GPU 上跑通过**（3/3 PASSED），且扩大后的 Scheduler/BlockManager/KVCache/CacheManager 回归范围（162 项，5 项需要真实模型正常 SKIPPED）全部保持通过，0 FAILED，证明 `read_block`/`write_block`/`write_blocks` 的改动未破坏任何现有非 chunked 路径。**这是本会话第一次真正验证 `use_chunked_kv_cache=true` 这条路径本身在 openvino.genai 层面的行为**（而不仅仅是编译通过）。仍未做的：`BlockManager`/`CacheOrchestrator` 层面"按整 chunk 增量取整"与调度循环的集成（目前只验证了 `KVCacheManager` 自身）；真实模型端到端生成测试；5c（跨 chunk 碎片整理 primitive，用于真正"收缩并归还显存"）完全未开始。 | `openvino/src/plugins/intel_gpu/include/intel_gpu/primitives/`（`paged_attention.hpp` 新增 `CHUNK_BASE_PTRS` 输入 + 新建碎片整理 primitive 定义）<br>`openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/`（kernel 实现，含 `paged_attention` 寻址公式改造）<br>`openvino/src/plugins/intel_gpu/tests/unit/test_cases/`（单元测试，含 5a 的独立技术验证实验）<br>`openvino.genai` 侧 `continuous_batching/cache/kv_cache_manager.hpp`（chunk 列表化改造，5d） | 5a：独立实验证明"USM 指针数组 kernel 参数、kernel 内解引用"在真实驱动上可行且性能可接受，这是继续往后所有子阶段的前提——**已验证**：`C:\Users\gta\phase5a_chunk_ptr_experiment\chunk_ptr_experiment.cpp`（独立、未纳入任何仓库构建系统的可抛弃程序）在真实 Intel Arc 140T GPU 上验证两个独立 `clDeviceMemAllocINTEL` chunk 通过一张 `clHostMemAllocINTEL` 指针表被同一次 kernel launch 正确解引用，8/8 block 读取正确，2000 次迭代平均 launch 2.39 us。5b/5c：新增/改造的 primitive 各自要有单元测试，比照 `pa_kv_reorder_gpu_test.cpp` 的写法，且"只有一个 chunk"时须与改造前行为逐位一致。5e：在 Qwen/Llama 32k 长文本、高并发场景下，用系统工具观察显存占用曲线验证收缩后真正归还给驱动，生成结果与不分 chunk 时逐字一致，并对比吞吐/TTFT 证明收益。 | Phase 4；且独立于 `openvino.genai` 侧其余工作，需要单独排期评审（跨仓库改动，风险和验证成本都更高）。 |

---

## 7. 风险评估与应对策略

1. **Host ↔ SSD I/O 阻塞主推理调度线程**：
   - *应对*：`TieredCacheManager` 内部维护独立的后台 I/O 线程池与环形队列，Swap-Out 过程完全异步化；Swap-In 采用预取管线（Prefetch Pipeline），在分词与调度阶段提前发出磁盘读请求。
2. **第三方 SSD 插件崩溃或内存泄漏**：
   - *应对*：在 `SSDPluginLoader` 中封装强防御边界，校验 C ABI 版本号；对插件返回的状态码进行严格检查。
     **插件是用户显式配置的（`storage_plugin_path` 非空）时，加载失败或运行时报错必须直接抛错终止，
     不允许静默切回 default backend**——静默回退会让用户误以为仍在使用厂商优化路径，掩盖真实的性能或
     正确性问题；只有当用户没有显式指定插件、系统自动探测到某个可选优化不可用时，才允许静默降级到
     default backend，并打印告警日志。
3. **模型换版 / Tokenizer 变更产生脏 Cache**：
   - *应对*：L2 存储 Manifest 强制绑定 Model Topology Fingerprint 与 Tokenizer Vocab Hash，指纹不一致时拒绝加载并原子清空历史存储。