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

### 0.3 全新工作，目前 0% 完成

**GPU Host↔Device 批量 Swap Kernel（本文档 §3.2.2 阶段 2）是当前最大的缺口，此前完全没有涉及**，需要
改到 `openvino`（GPU plugin）本体，而不是 `openvino.genai`。已核实的现状：

- `openvino` 的 GPU plugin 里已经存在一个方向相关、但**不能直接拿来用**的原语：
  `PA_KV_Reorder`（`src/plugins/intel_gpu/include/intel_gpu/primitives/pa_kv_reorder.hpp` +
  `src/graph/impls/ocl_v2/pa_kv_reorder.cpp` 和对应 `.cl` kernel）。它做的是**设备内部**、按
  `BLOCK_INDICES`/`BLOCK_UPDATE_INDICES` 批量 gather/scatter KV block（服务于 cache eviction 重排、
  压缩 KV cache、X-Attention），输入输出都是同一执行图里的设备 tensor，**不是** Host↔Device 传输，
  也不是一个可以在 model graph 之外被 `KVCacheManager` 直接调用的工具函数。
- 现有 `KVCacheManager::write_blocks()`（`openvino.genai` 侧，Phase 3 已完成）用的是
  `ov::RemoteTensor::copy_to/copy_from` 做 host↔device 批量 ROI 拷贝，这是 OpenVINO 已有的通用 Runtime
  API，不涉及 GPU plugin 内部改动，吞吐受限于这层通用拷贝路径，达不到本文档设想的"一个 Kernel Launch
  完成数十个分散 Block 并发搬移"的效果。
- 要真正做到本文档要求的效果，需要在 `openvino` GPU plugin 里新增一个 host↔device 感知的批量 swap
  primitive/kernel（可以参考 `pa_kv_reorder` 的 kernel 参数传递/block 索引处理方式，但要新增 host 侧
  buffer 的绑定和显式的 D2H/H2D 方向控制），工作量包含：primitive 定义、kernel_selector 注册、
  `.cl` kernel 实现、单元测试（`intel_gpu/tests/unit/test_cases/`）、真实 GPU 硬件验证。这是一项独立的、
  跨越两个仓库、有真实工程量的任务，不能和 `openvino.genai` 侧的其余工作混在一个 phase 里估算。

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
3. **GPU 推理后端（阶段 2 - 专用 PagedAttention 批量 Swap Kernel）**：
   - 在 `openvino` GPU plugin 中扩展算子/执行器：支持传入 `std::vector<size_t> src_block_ids, dst_block_ids`，一个 Kernel Launch 完成数十个分散 Block 的并发搬移。

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
| **Phase 4** | **新建** | **SSD Plugin 动态加载架构与 C ABI 封装**：目前没有任何 C ABI/动态加载代码。 | `src/cpp/include/openvino/genai/c/ssd_plugin_interface.h`<br>`src/cpp/src/continuous_batching/cache/ssd_plugin_loader.cpp` | 编写 Mock Vendor Plugin（`mock_ssd_plugin.so`），测试动态发现、载入、读写委托与异常故障隔离测试。（补充：插件显式配置但加载/运行失败时必须报错，不允许静默回退到 default backend，见 §0.2 对 §7 风险应对里"静默回退"提法的修正） | Phase 3 |
| **Phase 5** | **新建** | **GPU 运行时加速与端到端评测**：需要在 `openvino`（不是 `openvino.genai`）的 Intel GPU plugin 里新增一个 host↔device 感知的批量搬运 primitive/kernel（PagedAttention swap kernel 适配），目前 0% 完成。可参考同代码库已有的 `PA_KV_Reorder`（`src/plugins/intel_gpu/include/intel_gpu/primitives/pa_kv_reorder.hpp`）在 block 索引/kernel 参数传递上的做法，但 `PA_KV_Reorder` 本身是纯设备内 gather/scatter，不能直接复用，只能作为实现参考。 | `openvino/src/plugins/intel_gpu/include/intel_gpu/primitives/`（新 primitive 定义）<br>`openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/`（kernel 实现）<br>`openvino/src/plugins/intel_gpu/tests/unit/test_cases/`（单元测试） | 在 Qwen/Llama 32k 长文本场景下，对比开启 Offloading 前后的吞吐、并发容量与 TTFT 加速比。（补充：新增 primitive 需先有自己的单元测试，比照 `pa_kv_reorder_gpu_test.cpp` 的写法；并在真实 GPU 硬件上对比"通用 RemoteTensor ROI 拷贝路径"（Phase 2 已有）与"新 kernel 路径"的吞吐差异，证明确有收益才合入端到端评测） | Phase 4；且独立于 `openvino.genai` 侧其余工作，需要单独排期评审（跨仓库改动，风险和验证成本都更高）。 |

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