# OpenVINO GenAI 可扩容 PagedAttention

## 1. 背景

KV Cache会随着以下因素增长：

- prompt 越来越长；
- 生成 token 越来越多；
- 同时运行的请求越来越多。

当前传统实现中，每层 KV cache 是一个连续 tensor。假设已有 64 个 blocks，需求增长到 80 个 blocks：

```text
1. 分配新的 80-block tensor
2. 把旧 tensor 中的 64 个 blocks 全部复制过去
3. 把请求切换到新 tensor
4. 释放旧 tensor
```

这带来三个问题：

- 扩容需要复制已有数据，成本随 cache 大小增长；
- 新旧 tensor 同时存在时会产生显存峰值；
- 扩容可能阻塞一次 scheduler step，造成延迟尖峰。

本阶段的目标是：**让 GPU PagedAttention 支持非连续的 KV cache，并且扩容时只追加新显存，不复制已有 KV 数据。**


## 2. 可扩容 PagedAttention 的关键设计

### 2.1 传统连续 cache

```text
key_cache[layer]   = 一个连续 tensor
value_cache[layer] = 一个连续 tensor
```

block 通过连续偏移定位：

```text
address = cache_base + block_id * block_stride
```

扩容必须整体换 tensor，因此要复制旧数据。

### 2.2 Chunked cache

每层改为维护多个独立 allocation：

```text
key_chunks[layer]   = [chunk0, chunk1, chunk2, ...]
value_chunks[layer] = [chunk0, chunk1, chunk2, ...]
chunk_base_ptrs     = [chunk0地址, chunk1地址, ...]
```

其中：

```text
chunk0 / chunk1 / chunk2  -> 每个独立的 GPU USM allocation
chunk_base_ptrs           -> kernel 可读取的 chunk 首地址表
```

扩容时：

```text
旧状态: [chunk0, chunk1]
新增需求:
新状态: [chunk0, chunk1, chunk2]
```

只分配 `chunk2`，`chunk0` 和 `chunk1` 不动，已有 KV 数据不复制。

### 2.3 PagedAttention 如何定位一个 block

Chunked 模式的访问公式是：

```text
chunk_id      = block_id / blocks_per_chunk
local_block   = block_id % blocks_per_chunk
address       = chunk_base_ptrs[chunk_id] + local_block * block_size
```


这里的 `block_size` 表示一个 KV block 在对应 cache tensor 中的实际 stride；它包含该 block 的 token 数和 head/channel 维度布局，不只是一个字节数常量。

这就是“可扩容 PagedAttention”的核心：**逻辑 block id 保持连续，但物理 block 分布在多个 chunk 中。**


## 3. 一个完整的参数例子

假设 GPU 普通 PagedAttention 使用以下配置：

```python
scheduler_config.num_kv_blocks = 64
scheduler_config.max_num_batched_tokens = 512
scheduler_config.max_num_seqs = 2
scheduler_config.use_chunked_kv_cache = True
scheduler_config.kv_cache_chunk_size_blocks = 4

# 请求级配置
 generation_config.max_new_tokens = 32
```

### `block_size`：一个 block 能保存多少 token

普通 GPU PagedAttention 当前通常使用：

```text
block_size = 16 tokens
```

长度为 50 tokens 的请求需要：

```text
ceil(50 / 16) = 4 logical blocks
```

`block_size` 是 attention kernel 的基础布局参数，通常由设备和 kernel 路径决定，不是用户在普通 pipeline 配置中自由选择的参数。

### `num_kv_blocks`：初始逻辑容量

```text
64 blocks * 16 tokens = 1024 tokens
```

`num_kv_blocks=64` 表示 BlockManager 初始拥有 64 个逻辑 blocks。它不是每轮处理的 token 数，也不是单条请求的最大长度。

### `max_num_batched_tokens`：一轮最多处理多少 token

```text
max_num_batched_tokens = 512
```

它限制一次 scheduler step 的工作量。例如本轮有 550 个 prompt tokens，scheduler 可能拆成：

```text
本轮处理 512 tokens
下一轮处理剩余 38 tokens
```

它影响调度和 prefill 分批，不直接决定 KV cache 的总容量。

### `max_num_seqs`：一轮最多处理多少条 sequence

```text
max_num_seqs = 2
```

它限制一次 scheduler step 中参与计算的请求数量。它和 `max_num_batched_tokens` 共同限制调度规模：一个限制请求数，一个限制 token 数。

### `max_new_tokens`：请求未来可能需要多少 KV

```text
max_new_tokens = 32
```

scheduler 可以用 `prompt tokens + max_new_tokens` 估计该请求未来的 cache 需求。新请求动态到达时，总需求可能超过当前容量，从而触发扩容。

### `kv_cache_chunk_size_blocks`：物理增长粒度

```text
kv_cache_chunk_size_blocks = 4
```

结合 `block_size=16`：

```text
4 blocks/chunk * 16 tokens/block = 64 tokens/chunk
```

它决定每次追加多少物理显存。当前实现中，普通 GPU 的动态增长粒度常见为 256 tokens，因此测试中也使用过：

```text
16 blocks/chunk * 16 tokens/block = 256 tokens/chunk
```

chunk size 越小，增长更细但 allocation 和 pointer table 管理成本更高；越大，管理成本更低但显存粒度更粗。


### 实验结果：局部扩容收益与整体 regression

使用真实 Qwen3-8B、Intel Arc 140T GPU，构造两步场景：

```text
Step 0: Wave 1 建立已有 KV cache
Step 1: Wave 2 小请求到达，触发 cache 扩容
```

一次代表性结果如下：

| 指标 | Non-chunked | Chunked | 结果 |
|---|---:|---:|---|
| Step 0：建立 Wave 1 cache | 1634.39 ms | 1890.94 ms | Chunked 慢 15.7% |
| Step 1：扩容 + 小请求 prefill | 581.34 ms | 518.62 ms | Chunked 快 10.8% |
| 两个 step 合计 | 2215.73 ms | 2409.56 ms | Chunked 慢 8.7% |

这说明两件事：

1. **扩容动作本身有收益**：Chunked 不需要复制已有 KV，因此扩容 step 更快。
2. **当前整体仍有 regression**：Step 0 和后续稳态计算都要为 `chunk_base_ptrs` 间接寻址付费，这个固定开销超过了单次扩容节省的时间。

在固定容量、无扩容的稳态测试中，chunked 相比 non-chunked 约慢 11-12%；缓存 `clSetKernelExecInfo` 的 pointer set 后有小幅改善，但无法消除 kernel 侧的间接寻址成本。

因此当前阶段的准确结论是：

> Chunked KV cache 已验证“扩容时不复制旧 KV”的局部收益，但尚未证明整体吞吐收益；当前典型 workload 下存在稳态性能 regression。它首先是显存弹性和扩容稳定性的基础设施，后续需要继续优化 kernel 寻址，或在更高扩容压力的 workload 中评估收益是否能够覆盖稳态成本。

## 4. 当前限制和下一步

当前只实现了 append-only 增长：

```text
可以增长:  [chunk0, chunk1] -> [chunk0, chunk1, chunk2]
还不能完整收缩: 释放空 chunk 并把显存归还给 driver
```

下一步是 Phase 5c：

1. 实现跨 chunk 的 block move/defragmentation；
2. 把仍在使用的 blocks 搬出待释放 chunk；
3. 销毁空 chunk，真正归还 GPU 显存；
4. 再验证长上下文、高并发和显存紧张场景。

