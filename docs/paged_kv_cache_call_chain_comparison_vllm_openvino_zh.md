# OpenVINO GenAI、OpenVINO Runtime 与 vLLM 的 Paged KV Cache 调用链对比

本文从调用链角度对比三层：

```text
OpenVINO GenAI：连续批处理、request 生命周期、logical -> physical block
OpenVINO Runtime：CompiledModel / InferRequest / PagedAttention 图执行
vLLM V1：Scheduler、KVCacheManager、attention backend，以及 KV offload connector
```

重点是帮助实现者回答三个问题：

1. paged KV cache 的 block 是在哪里分配和释放的？
2. physical block table 是在哪里转换为模型输入的？
3. 如果加入 CPU/NVMe offload，store/load 应该挂在哪个边界？

## 1. 一句话结论

两边的共同核心是：

```text
token sequence
  -> logical blocks
  -> block table
  -> physical block IDs
  -> paged attention 根据 ID 访问 KV cache page
```

主要差异是：

| 维度 | OpenVINO GenAI | vLLM V1 |
|---|---|---|
| 运行时形态 | C++ pipeline 直接持有 `ov::InferRequest` | Python scheduler/worker 协作，模型执行由 vLLM engine 驱动 |
| 物理 cache | `KVCacheManager` 持有每层 `ov::Tensor` / `ov::RemoteTensor` | attention backend 注册的 GPU KV tensors，由 worker 访问 |
| block 管理 | `BlockManager` + `BlockAllocator` + `BlocksPerLayer` | `KVCacheManager` / `BlockPool`，按 KV cache group 管理 |
| per-layer 关系 | 一个逻辑 block 以 `BlocksPerLayer` 跨层成组 | 由 KV cache groups / layer storage 描述，布局由 worker 解析 |
| prefix cache | `hash -> BlocksPerLayer`，另有 `OverwritableBlocksHashStore` | `block_hash -> block`，CPU offload 还有独立 offload block pool |
| block table 输入 | `ModelRunner::_fill_indices_from_block_tables()` 填 `block_indices(.N)` | scheduler 输出 block IDs，attention metadata / model runner 传给 backend |
| eviction | OpenVINO GenAI 在 `infer()` 后根据 attention score 主动 eviction | 常规 block 回收由 scheduler / BlockPool 管理；offload 可按 eager/lazy store |
| offload 通信 | 适合在 `CacheOrchestrator` / pipeline step 内加入 C++ transfer manager | `SimpleCPUOffloadConnector` 分成 scheduler manager、worker handler、disk backend |
| GPU 同步 | 需要确认 `RemoteTensor` copy 与 plugin queue 语义 | CUDA stream + CUDA event 已明确建模 |

最重要的边界判断：OpenVINO 不能把 vLLM 的 `cpu_block_id` 直接照搬为 `physical_block_id`。OpenVINO 需要明确区分：

```text
OpenVINO physical block ID  -> key_cache/value_cache tensor 中的第 0 维
OpenVINO disk slot ID       -> offload 文件中的固定 slot
```

## 2. 两边的总调用链

### 2.1 OpenVINO GenAI

```text
ContinuousBatchingImpl 构造
  -> prepare_model_for_paged_attention()
  -> compile_model()
  -> create_infer_request()
  -> CacheOrchestrator::create()
       -> detect_cache_managers()
       -> KVCacheManager
       -> BlockManager
  -> Scheduler
  -> ModelRunner

每个 step：

_pull_awaiting_requests()
  -> Scheduler::schedule()
       -> BlockManager 分配 token blocks
       -> CacheOrchestrator::allocate_cache_if_needed()
       -> KVCacheManager 创建/扩展 physical KV tensors
       -> BlockManager::append_slots()
       -> CacheOrchestrator::copy_blocks()  // fork/COW
       -> Scheduler::Output
  -> ModelRunner::forward()
       -> 填 block_indices / past_lens / 输入 token
       -> InferRequest::infer()
  -> _maybe_evict_cache_blocks()
       -> attention scores
       -> logical block eviction
       -> free_blocks_from_sequence()
  -> sampler
       -> fork_sequence() / free_sequence()
```

### 2.2 vLLM V1 普通 paged KV cache

```text
VllmConfig / CacheConfig
  -> KVCacheConfig / KV cache groups
  -> KVCacheManager / BlockPool
  -> Scheduler
  -> scheduler_output
  -> worker model execution
       -> attention metadata
       -> block table / slot mapping
       -> model forward
       -> KV cache kernels 写入 physical pages
  -> request 完成 / preemption
       -> block ref count 下降
       -> free queue / prefix cache
```

具体类名会随 vLLM 版本和 V0/V1 路径变化；下面 offload 对比使用当前仓库的 V1 `SimpleCPUOffloadConnector` 实现。

### 2.3 vLLM V1 磁盘 offload

```text
KVTransferConfig
  -> SimpleCPUOffloadConnector
       role=SCHEDULER -> SimpleCPUOffloadScheduler
       role=WORKER    -> SimpleCPUOffloadWorker

Scheduler side:
  -> get_num_new_matched_tokens()
  -> update_state_after_alloc()
  -> prepare_store_specs()
  -> build_connector_meta()
       load/store block IDs + event IDs

Worker side:
  -> bind_connector_metadata()
  -> get_finished()
       store waits compute-done CUDA event
       load/store launch_copy()
       poll CUDA events
  -> build_connector_worker_meta()

I/O side:
  -> DiskBackend store/load queue
  -> pinned staging buffer
  -> GPU DMA
  -> pwritev/preadv
  -> fixed disk slot
```

## 3. 初始化阶段对比

### 3.1 OpenVINO GenAI 初始化

OpenVINO GenAI 先修改模型图，再编译：

```cpp
prepare_model_for_paged_attention(model, scheduler_config);
ov::CompiledModel compiled_model =
    utils::singleton_core().compile_model(model, device, properties);
ov::InferRequest infer_request = compiled_model.create_infer_request();
```

接着 `CacheOrchestrator::create()` 从 compiled model 的输入名识别：

```text
key_cache.0, key_cache.1, ...
value_cache.0, value_cache.1, ...
```

`KVCacheManager` 从输入 shape 和 precision 推导：

- decoder layer 数。
- block size。
- 每个 layer 的 key/value shape。
- 一个跨所有 layer 的 block bytes。
- CPU 或 GPU/RemoteTensor 分配方式。

然后 `CacheOrchestrator` 创建对应的 `BlockManager`，并把 block count 写回 scheduler config。

### 3.2 vLLM 初始化

vLLM 先从配置得到 KV cache config，再由 connector factory 创建 KV connector。普通 paged KV cache 的物理 tensor 会由 worker/attention backend 创建；offload connector 额外创建 scheduler-side 和 worker-side 两个角色对象。

磁盘模式下：

```text
worker.register_kv_caches(kv_caches)
  -> 去重底层 storage
  -> 将每个 storage 映射为 [num_blocks, block_bytes]
  -> 计算 bytes_per_block
  -> DiskBackend.init()
       -> 文件和 slot
       -> store/load staging buffers
       -> store/load CUDA streams
       -> store/load 后台线程
```

### 3.3 初始化设计差异

OpenVINO 的物理 cache tensor 是 `KVCacheManager` 的私有成员，并绑定到一个 `InferRequest`；vLLM 的 worker 可以注册 attention backend 提供的多种 KV tensor layout，再构造通用 copy 参数。

因此迁移到 OpenVINO 时，不能只新增一个文件后端。必须给 `KVCacheManager` 暴露一个稳定的“按 layer/key/value/block 读写”接口，或者由 offload manager 持有受控的 `KVCacheManager` 引用。

## 4. Block 分配和物理存储对比

### 4.1 OpenVINO 的分配模型

`BlockAllocator` 为每层维护 free list。`BlockManager` 维护：

```text
m_block_table[seq_id][layer][logical_block_idx]
```

每个元素是 `CacheBlock`，其 `get_index()` 是 physical block ID。跨层同一个逻辑 block 通过 `BlocksPerLayer` 关联。

分配时：

```text
BlockManager::allocate_tokens()
  -> allocate()
       -> allocate_cached_block(hash, content_length)
          或 BlockAllocator::allocate_block()
  -> block table 增加 physical blocks
```

physical block ID 直接对应 `m_key_cache[layer][physical_id]` 和 `m_value_cache[layer][physical_id]` 的第 0 维。

### 4.2 vLLM 的分配模型

vLLM 的 block manager 以 KV cache group 为单位管理 block。block ID 也最终是 attention backend 访问 KV tensor 的 page index，但不同 backend 可能有不同的物理 layout。

Simple CPU offload scheduler 会派生一个 offload KV cache config：

```text
GPU KV capacity + offload capacity
  -> CPU/offload BlockPool
```

磁盘模式下 offload pool 的 block ID 就是 disk slot ID。这个 pool 只负责 hash、free queue、引用计数和 slot 分配；真实文件读写在 DiskBackend。

### 4.3 不能直接类比的点

| vLLM 概念 | OpenVINO 对应物 | 是否可直接复用 |
|---|---|---|
| GPU block ID | `CacheBlock::get_index()` | 语义相近 |
| CPU/offload block ID | 不应直接复用；应新增 disk slot ID | 不可直接复用 |
| `BlocksPerLayer` | OpenVINO 已有 `BlocksPerLayer` | 可复用 |
| BlockPool free queue | `BlockAllocator::m_free_blocks` + overwrite store | 需适配 |
| block hash map | `m_prefix_hash_to_cached_blocks` | 可扩展 |
| `touch()` | `CacheBlock::increment()` / transfer pin | 需明确区分 owner 与 transfer pin |
| KV group layout | 每层 key/value tensor shape | 需由 KVCacheManager 统一描述 |

## 5. 一次 token 写入的调用链

### 5.1 OpenVINO

```text
Scheduler::schedule()
  -> 为序列分配/追加 physical blocks
  -> Scheduler::Output 保存 block tables
  -> ModelRunner::forward()
       -> _fill_indices_from_block_tables()
       -> block_indices(.N)
       -> past_lens / subsequence_begins
       -> m_request.infer()
            -> PagedAttention 图
                 -> 根据 block_indices 读取已有 KV
                 -> 将当前 token 的 K/V 写入对应 physical block
```

OpenVINO 的 `ModelRunner` 不负责把 KV 写入 C++ cache tensor；它负责把 tensor 和索引作为 model input 交给编译后的图。KV 写入由 PagedAttention 图和 device plugin 执行。

### 5.2 vLLM

```text
Scheduler
  -> scheduler_output
  -> worker/model runner
       -> attention metadata
       -> block table / slot mapping
       -> model forward
            -> attention backend kernel
                 -> 读取历史 KV pages
                 -> 写入当前 token KV page
```

vLLM 的 KV write 也由 attention kernel 完成，但 worker 可以在 forward 之后用 CUDA stream/event 对 GPU KV storage 做 DMA copy。

### 5.3 对 offload 的影响

两者都必须满足：

```text
compute 写 KV 完成
  -> 才能 store
```

vLLM 已经用 `store_compute_done` CUDA event 表达这个依赖。OpenVINO 第一阶段可以把 `ModelRunner::forward()` 返回作为 CPU 模式 store 的同步边界；GPU RemoteTensor 模式需要额外确认 plugin/device 到 host 的可见性。

## 6. Prefix cache restore 对比

### 6.1 OpenVINO

新 request 在 `add_request()` 中调用：

```text
m_scheduler->restore_cached_blocks(sequence_group)
  -> CacheOrchestrator::restore_cached_blocks()
  -> BlockManager::get_prefix_restore_plan()
  -> BlockAllocator::get_cached_block(hash, cached_map)
  -> 把已有 BlocksPerLayer 放入 m_block_table
  -> update_processed_tokens_num()
```

当前 restore 是同步的逻辑操作：命中 block 已经在 OpenVINO physical cache 中，因此可以直接挂入 block table。

### 6.2 vLLM

Simple CPU offload scheduler 在新 request 调度时：

```text
get_num_new_matched_tokens()
  -> cpu_coordinator.find_longest_cache_hit()
  -> touch CPU blocks
  -> 保存 _pending_cpu_hits

update_state_after_alloc()
  -> 分配 GPU destination blocks
  -> 建立 CPU slot -> GPU block pairs
  -> touch source/destination

build_connector_meta()
  -> 分配 load_event
  -> 把 load request 发给 worker

DiskBackend
  -> preadv disk slot
  -> staging buffer -> GPU DMA

worker event 完成
  -> finished_recving(request IDs)
  -> scheduler cleanup load state
```

### 6.3 OpenVINO 迁移后的必要变化

OpenVINO 现有 `restore_cached_blocks()` 不能直接把 disk source 放进 block table，因为 ModelRunner 会马上把 physical ID 交给 PagedAttention，而该 physical block 可能还没有恢复数据。

建议把 restore 拆成：

```text
get_prefix_restore_plan()
  -> 返回 source = DEVICE / OVERWRITEABLE / DISK

allocate_restore_destinations()
  -> 分配 device physical blocks

start_load()
  -> disk slot -> destination physical blocks

wait_for_required_loads()
  -> load 完成

commit_restore()
  -> 正式写入 block table
  -> 更新 processed tokens
```

这也是两边最重要的调用链差异之一。

## 7. Eviction、free 和 reuse 对比

### 7.1 OpenVINO eviction

OpenVINO GenAI 的 `_maybe_evict_cache_blocks()` 在当次 `infer()` 完成后执行：

```text
ModelRunner::forward()
  -> m_model_runner->get_last_attention_scores()
  -> CacheEvictionAlgorithm::register_new_token_scores()
  -> evict_logical_blocks()
  -> Scheduler::free_blocks_from_sequence()
  -> BlockManager::free_blocks_from_sequence()
  -> free_cached_blocks()
```

它按 attention score 选择 logical block，再转换到对应 physical `CacheBlock`。如果 prefix cache 打开，free 后可能进入 `OverwritableBlocksHashStore`。

### 7.2 vLLM eager/lazy offload

vLLM 有两类 store 选择：

```text
eager:
  _prepare_eager_store_specs()
    -> 按 request 收集已经 confirmed 的新 block

lazy:
  _prepare_lazy_store_specs()
    -> 扫描 GPU free queue
    -> 选择接近被回收、但仍有 hash 的 block
```

store 完成之前，GPU block 会被 touch/in-flight 集合保护；完成之后才把 offload block 的 hash 放进 CPU prefix cache map。

### 7.3 OpenVINO 推荐的第一版

建议先做：

```text
infer 返回
  -> eviction 选出 physical BlocksPerLayer
  -> 在 BlockManager 释放前捕获它们
  -> store pin
  -> enqueue disk store
  -> store 完成后发布 disk hash
  -> 允许原 physical block 真正复用
```

不建议第一版直接实现 vLLM lazy cursor，因为 OpenVINO 已经有基于 attention score 的 eviction 触发点，先复用现有策略更容易验证正确性。

## 8. Copy-on-write、fork 和 preemption 对比

### 8.1 Copy-on-write / fork

OpenVINO：

```text
BlockManager::fork_sequence()
  -> CacheBlock::increment()

BlockManager::append_slots()
  -> 检查 last_block->copy_on_write()
  -> 分配 destination block
  -> 返回 source physical -> destination physical map

CacheOrchestrator::copy_blocks()
  -> KVCacheManager::copy_blocks()
```

vLLM：

```text
fork / beam / scheduler allocation
  -> block ref count 增加
  -> COW 时 attention/block manager 产生 copy pairs
  -> worker/model backend 执行 block copy
```

如果 OpenVINO 的 source block 已经在 disk，COW 不能直接调用普通 `KVCacheManager::copy_blocks()`。必须：

```text
disk source
  -> restore 到 temporary/device source
  -> device source -> COW destination
  -> 释放 temporary source
```

第一版可以限制为：disk source 必须先完整 restore，再允许 fork/COW。

### 8.2 Preemption/reset

vLLM connector 在 preemption 时会 flush 所有 in-flight CUDA events，避免被 allocator 复用的 block 被旧 DMA 覆盖；reset 时保留 abandoned transfer metadata，完成后只释放引用，不把 stale store 重新发布成 cache。

OpenVINO 目前主要通过 scheduler 的 recompute preemption/free sequence 释放 blocks。加入 offload 后必须补同等语义：

```text
preempt/free
  -> 如果 transfer in-flight，不立即复用物理 block
  -> flush 或等待 transfer
  -> load/store record 完成后清理
```

## 9. Store/load 插入点并排比较

| 阶段 | OpenVINO 推荐位置 | vLLM 当前位置 | 设计原因 |
|---|---|---|---|
| store 候选选择 | `_maybe_evict_cache_blocks()` / `BlockManager::free_blocks_from_sequence()` 前 | `prepare_store_specs()` | 必须拿到真实 physical block |
| store 提交 | `infer()` 返回后，pipeline step 内 | worker `get_finished()` | 避免读到 compute 未完成的 KV |
| store I/O | 新增 C++ transfer manager | `DiskBackend._store_loop()` / `_do_store()` | 后台线程 + staging buffer |
| store publish | disk write/copy event 完成后 | `_process_store_event()` | 完成前不可 prefix hit |
| load 发现 | `BlockManager` prefix restore plan 扩展 disk source | `get_num_new_matched_tokens()` | 按 hash 找最长连续 prefix |
| load 分配目标 | restore plan 后分配 device blocks | `update_state_after_alloc()` | source slot 与 destination block 成对 |
| load 提交 | 下一次 `ModelRunner::forward()` 前 | `get_finished()` 提交后异步完成 | attention 不能读未恢复 block |
| load 完成 | commit block table 后才更新 processed tokens | `finished_recving` 唤醒 request | 保证 scheduler/model 一致 |
| block reuse | store/load pin 释放后 | CPU/GPU block ref 清理后 | 防止旧 transfer 覆盖新 owner |

## 10. 关键状态映射

### 10.1 OpenVINO 建议状态

```text
DEVICE_ACTIVE
  -> STORE_PINNED
  -> STORE_IN_FLIGHT
  -> DISK_CACHED

DISK_CACHED
  -> LOAD_DEST_ALLOCATED
  -> LOAD_IN_FLIGHT
  -> DEVICE_ACTIVE
```

### 10.2 vLLM 当前状态

```text
GPU block / CPU offload block
  -> store event
  -> backend queue
  -> CUDA event
  -> completed store metadata
  -> CPU hash cache published

CPU hit
  -> pending CPU hits
  -> GPU destination allocation
  -> load event
  -> finished request IDs
  -> cleanup touch refs
```

### 10.3 同步不可违反的约束

```text
STORE_IN_FLIGHT 的 physical block 不可进入 allocator 可复用池
LOAD_IN_FLIGHT 的 destination block 不可交给 PagedAttention
STORE 未完成的 hash 不可加入 prefix cache
reset/preemption 不能丢弃 transfer record
同一个 BlocksPerLayer 的所有 decoder layer 必须一致地完成状态迁移
```

## 11. 对 OpenVINO Runtime 的特殊注意事项

### 11.1 `ov::Tensor` 与 `ov::RemoteTensor`

CPU 路径中，KV tensor 通常可通过 `data()`、ROI 和 `copy_to()` 读写。GPU 路径中，`KVCacheManager` 使用 `RemoteContext::create_tensor()` 创建 RemoteTensor；它不是普通 host pointer。

因此：

- 不要用 `reinterpret_cast<uint8_t*>(remote_tensor.data())` 假设能直接读 GPU 内容。
- 不要把 `KVCacheManager::copy_blocks()` 中 remote tensor 的行为当作通用 GPU copy API；当前代码在 remote 情况下会跳过 host `memcpy`。
- GPU disk offload 需要显式验证 RemoteTensor 到 host staging 的 copy、完成时机和 plugin queue 同步。

推荐验证顺序：

```text
CPU + synchronous disk
  -> CPU + background I/O
  -> GPU + synchronous RemoteTensor/host copy
  -> GPU + background I/O
  -> O_DIRECT / multi-buffer optimization
```

### 11.2 OpenVINO Runtime 与 vLLM CUDA 的边界差异

| 能力 | OpenVINO | vLLM |
|---|---|---|
| 设备 tensor | `ov::RemoteTensor` | `torch.Tensor` on CUDA/ROCm |
| 异步执行表达 | plugin/runtime 内部 queue，需确认 API 语义 | `torch.cuda.Stream` / `torch.Event` |
| host pinning | 由 Runtime/plugin 或 host allocator 能力决定 | `cudaHostRegister` / pinned tensor |
| batch DMA | 需使用 OpenVINO/driver 支持的 copy API | `cuMemcpyBatchAsync` / `hipMemcpyBatchAsync` |
| event polling | 需要定义 C++ future/event 或同步点 | CUDA event `query()` |
| 文件 I/O | 可复用固定 slot + `pread/pwrite` | 当前 DiskBackend 使用 `preadv/pwritev` |

## 12. 迁移时最容易混淆的三个地址空间

```text
逻辑 block index
  - sequence 内的第几个 token block
  - eviction algorithm 操作它

physical block ID
  - OpenVINO key/value cache tensor 第 0 维
  - block table 和 ModelRunner 使用它

 disk slot ID
  - offload 文件的第几个固定槽位
  - 只由 disk backend 使用
```

例子：

```text
seq logical block 3
  -> OpenVINO physical block 11
  -> disk slot 5

forward 时 block_indices 填 11
store/load 时文件偏移使用 5 * slot_bytes
```

如果把 5 填入 `block_indices`，Paged Attention 会访问错误的 GPU page；如果把 11 当文件 slot，磁盘索引会与 allocator 的 slot 生命周期混淆。

## 13. 推荐阅读路径

### 13.1 先读 OpenVINO

1. [docs/paged_kv_cache_call_chain.md](paged_kv_cache_call_chain.md)：现有 paged KV cache 总链路。
2. [src/cpp/src/continuous_batching/pipeline_impl.cpp](../src/cpp/src/continuous_batching/pipeline_impl.cpp)：初始化、`step()`、infer 后 eviction。
3. [src/cpp/src/continuous_batching/cache/cache_orchestrator.hpp](../src/cpp/src/continuous_batching/cache/cache_orchestrator.hpp)：cache manager/block manager 路由。
4. [src/cpp/src/continuous_batching/cache/block_manager.hpp](../src/cpp/src/continuous_batching/cache/block_manager.hpp)：logical/physical table、free、prefix restore、COW。
5. [src/cpp/src/continuous_batching/cache/kv_cache_manager.hpp](../src/cpp/src/continuous_batching/cache/kv_cache_manager.hpp)：key/value tensor 和 physical block copy。
6. [src/cpp/src/continuous_batching/scheduler.hpp](../src/cpp/src/continuous_batching/scheduler.hpp)：schedule 输出和 preemption API。
7. [src/cpp/src/continuous_batching/model_runner.hpp](../src/cpp/src/continuous_batching/model_runner.hpp)：block table 到 model input 的转换。
8. [src/cpp/src/continuous_batching/cache/cache_eviction.cpp](../src/cpp/src/continuous_batching/cache/cache_eviction.cpp)：logical block eviction。

### 13.2 再读 vLLM offload

1. [vllm/v1/simple_kv_offload/manager.py](../../vllm/vllm/v1/simple_kv_offload/manager.py)：scheduler 侧命中、store/load 准备和完成回收。
2. [vllm/v1/simple_kv_offload/metadata.py](../../vllm/vllm/v1/simple_kv_offload/metadata.py)：event metadata。
3. [vllm/v1/simple_kv_offload/worker.py](../../vllm/vllm/v1/simple_kv_offload/worker.py)：stream、event 和 transfer submit。
4. [vllm/v1/simple_kv_offload/disk_backend.py](../../vllm/vllm/v1/simple_kv_offload/disk_backend.py)：固定 slot、staging buffer、后台 I/O。
5. [vllm/v1/simple_kv_offload/cuda_mem_ops.py](../../vllm/vllm/v1/simple_kv_offload/cuda_mem_ops.py)：block bytes 和批量 DMA 地址计算。
6. [docs/paged_kv_cache_offload_migration_spec_zh.md](paged_kv_cache_offload_migration_spec_zh.md)：OpenVINO 迁移规格。

## 14. 对 OpenVINO offload 实现的直接建议

### 第一阶段：正确性优先

```text
CPU device
+ prefix caching
+ eviction-driven eager store
+ fixed disk slots
+ synchronous load before forward
+ no O_DIRECT
+ no GPU RemoteTensor
```

目标是验证：

- store 后重新 request 能命中相同 prefix。
- load 后 token 输出与无 offload baseline 一致。
- 多 layer key/value 内容一致。
- disk slot 与 physical block 不混淆。
- block 不会在 transfer 未完成时被复用。

### 第二阶段：状态机和后台 I/O

```text
独立 store/load queue
pinned staging buffer
monotonic event ID
preemption/reset flush
in-flight block pin
```

### 第三阶段：GPU 和性能

```text
RemoteTensor -> host staging
host staging -> disk
disk -> host staging -> RemoteTensor
multi-buffer pipeline
O_DIRECT / alignment
```

## 附录：源码索引

### OpenVINO GenAI

- [pipeline_impl.cpp](../src/cpp/src/continuous_batching/pipeline_impl.cpp)：pipeline 初始化、`step()`、`infer()` 后 eviction。
- [pipeline_impl.hpp](../src/cpp/src/continuous_batching/pipeline_impl.hpp)：Continuous Batching pipeline 声明。
- [scheduler.hpp](../src/cpp/src/continuous_batching/scheduler.hpp)：调度、输出 block table、preemption/free API。
- [model_runner.hpp](../src/cpp/src/continuous_batching/model_runner.hpp)：填充 `block_indices`、调用 `InferRequest`。
- [cache_orchestrator.hpp](../src/cpp/src/continuous_batching/cache/cache_orchestrator.hpp)：cache type、physical manager、block manager 的统一路由。
- [block_manager.hpp](../src/cpp/src/continuous_batching/cache/block_manager.hpp)：`CacheBlock`、`BlockAllocator`、prefix cache、COW、restore、free。
- [kv_cache_manager.hpp](../src/cpp/src/continuous_batching/cache/kv_cache_manager.hpp)：per-layer key/value tensors 和 physical block copy。
- [i_cache_manager.hpp](../src/cpp/src/continuous_batching/cache/i_cache_manager.hpp)：physical cache manager 抽象。
- [cache_eviction.cpp](../src/cpp/src/continuous_batching/cache/cache_eviction.cpp)：attention score 到 logical block eviction。

### vLLM

- [simple_cpu_offload_connector.py](../../vllm/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py)：connector 配置和 scheduler/worker 分叉。
- [manager.py](../../vllm/vllm/v1/simple_kv_offload/manager.py)：scheduler 侧 block/offload 状态机。
- [metadata.py](../../vllm/vllm/v1/simple_kv_offload/metadata.py)：transfer event metadata。
- [worker.py](../../vllm/vllm/v1/simple_kv_offload/worker.py)：GPU KV layout、stream、event、transfer 提交。
- [disk_backend.py](../../vllm/vllm/v1/simple_kv_offload/disk_backend.py)：磁盘固定 slot 和异步 I/O。
- [cuda_mem_ops.py](../../vllm/vllm/v1/simple_kv_offload/cuda_mem_ops.py)：CUDA/HIP batch memcpy。
- [test_scheduler.py](../../vllm/tests/v1/simple_kv_offload/test_scheduler.py)：scheduler store/load、lazy/eager、LRU、preemption 测试。
- [test_worker.py](../../vllm/tests/v1/simple_kv_offload/test_worker.py)：compute-done event 和 worker event 测试。
