# DFlash Step Implementation 说明

## 概述

本文档详细说明了 `dflash_strategy.cpp` 中 `step()` 函数的实现，该实现基于 DFlash 论文算法，并针对 continuous batching + paged attention 后端进行了优化。

参考实现：https://github.com/z-lab/dflash/blob/main/dflash/model.py#L63

## DFlash 算法核心流程

DFlash 是一种基于 draft model 的推测解码算法，其核心思想是：

1. **Draft Virtual Prefill**: Draft model 做一次"虚拟预填充"来分配 KV cache blocks (仅首次或新请求)
2. **Draft Model Generation**: Draft model 基于 main model 的 hidden states，快速生成多个候选 tokens (block_size 个)
3. **Main Model Verification**: Main model 在**单次** forward pass 中并行验证所有候选 tokens
4. **Accept/Reject & Update**: 接受匹配的 tokens，更新两个模型的状态

**关键**: Main model 每个 step 只运行**一次**，不是两次！

## 实现细节

### Step 1: Draft Model Virtual Prefill (首次或新请求)

**为什么需要？**

在 paged attention 中，KV cache 按 block 分配。Draft model 在生成候选 tokens 之前，必须：
1. **分配 KV cache blocks**: 预留足够空间存储 block_size 个 tokens 的 KV
2. **初始化 position_ids**: 确保位置编码从正确的位置开始
3. **接收 hidden states**: 从 main model 获取 context

**实现**:

```cpp
// Check which requests need draft virtual prefill
auto main_sequences_for_init = m_main_pipeline->get_generated_requests();

for (const auto& main_req : main_sequences_for_init) {
    auto request_id = main_req.first;
    const auto& main_sequences = main_req.second;

    if (draft_model_needs_init_for_this_request) {
        // Step 1a: Update draft state with main's hidden states
        for (const auto& [seq_id, generated_seq] : main_sequences) {
            if (generated_seq.hidden_states) {
                GeneratedSequences draft_init_seq;
                draft_init_seq[seq_id] = GeneratedSequence(
                    generated_seq.token_ids,
                    generated_seq.log_probs,
                    generated_seq.hidden_states
                );
                
                // is_prefill=true: prepare for KV allocation
                m_draft_pipeline->update_request(request_id, draft_init_seq, true);
            }
        }

        // Step 1b: Actually call draft step() to allocate KV cache blocks!
        m_draft_pipeline->step();
    }
}
```

**关键点**:
1. `update_request(..., true)` 更新 draft 的内部状态（token_ids, hidden_states, position_ids）
2. `m_draft_pipeline->step()` **实际执行**，触发 paged attention 分配 KV cache blocks
3. 这个 step 会运行 draft model inference，但主要目的是初始化 KV cache

**对应 Python 实现**:
```python
# dflash/model.py line ~115
if block_size > 1:
    noise_embedding = target.model.embed_tokens(block_output_ids)
    draft_logits = target.lm_head(model(
        target_hidden=target_hidden,  # <- Main model 的 hidden states
        noise_embedding=noise_embedding,
        position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
        past_key_values=past_key_values_draft,  # <- Draft model 的 KV cache
        use_cache=True,
        is_causal=False,
    )[:, 1 - block_size :, :])
    past_key_values_draft.crop(start)  # <- Crop KV cache
```

### Step 2: Draft Model Speculative Generation

```cpp
const auto draft_start = std::chrono::steady_clock::now();
m_draft_pipeline->multistep();
const auto draft_end = std::chrono::steady_clock::now();

auto draft_generated_requests = m_draft_pipeline->get_generated_requests();
```

**目的**:
- Draft model 基于 main model 的 hidden states 生成候选 tokens
- 一次生成 `num_assistant_tokens` (即 block_size) 个 tokens

**Draft Model 的输入**:
1. **target_hidden**: Main model 的中间层 hidden states (通过 `set_hidden_state_import_needed(true)` 导入)
2. **noise_embedding**: Draft model 自己的 token embeddings
3. **position_ids**: 正确的位置编码
4. **past_key_values**: Draft model 的 KV cache

**关键点**:
- Draft model 使用 **non-causal attention** (`is_causal=False`)
- Draft model 的 attention 同时 attend 到：
  - `target_hidden` (main model context)
  - 自己生成的 tokens (noise_embedding)

### Step 3: Insert Draft Tokens into Main Model

```cpp
for (const auto& draft_request : draft_generated_requests) {
    auto request_id = draft_request.first;
    const auto& draft_sequences = draft_request.second;

    // is_prefill=false: incremental insertion
    auto insert_result = m_main_pipeline->update_request(request_id, draft_sequences, false);

    update_sequence_info[request_id].inserted_tokens_cnt += insert_result.inserted_tokens_cnt;
}
```

**目的**:
- 将 draft tokens 插入 main model 的 KV cache
- 准备 main model 的并行验证

**is_prefill=false 的含义**:
- 增量插入 tokens，不重新分配 blocks
- 更新 position_ids (连续递增)
- 扩展 KV cache (但不运行推理)

### Step 4: Main Model Verification Pass (唯一的 Main Step!)

```cpp
const auto verify_start = std::chrono::steady_clock::now();
m_main_pipeline->step();
const auto verify_end = std::chrono::steady_clock::now();

auto verified_requests = m_main_pipeline->get_generated_requests();
```

**目的**:
- Main model 并行验证 draft tokens
- 运行 forward pass，计算每个位置的 logits
- 比较 draft tokens 与 main model 的预测

**并行验证原理**:
```
输入序列: [a, b1, b2, b3, b4]  (draft tokens)
Main model attention:
  pos 0: [] -> a
  pos 1: [a] -> b1
  pos 2: [a, b1] -> b2
  pos 3: [a, b1, b2] -> b3
  pos 4: [a, b1, b2, b3] -> b4

比较: draft tokens vs main predictions
  b1 == pred(a) ?      -> accept/reject
  b2 == pred(a, b1) ?  -> accept/reject
  ...
```

**对应 Python 实现**:
```python
# dflash/model.py line ~127
output = target(
    block_output_ids,  # <- Draft tokens
    position_ids=block_position_ids,
    past_key_values=past_key_values_target,
    use_cache=True,
    output_hidden_states=block_size > 1,
)

posterior = sample(output.logits, temperature)
acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
```

### Step 6: Update Draft Model based on Verification

```cpp
for (const auto& verified_request : verified_requests) {
    auto request_id = verified_request.first;
    const auto& verified_sequences = verified_request.second;

    // is_prefill=true: crop KV cache
    auto update_result = m_draft_pipeline->update_request(request_id, verified_sequences, true);
    update_sequence_info[request_id].removed_tokens_cnt = update_result.removed_tokens_cnt;
}
```

**目的**:
- 根据验证结果更新 draft model
- **Crop KV cache**: 移除被拒绝的 tokens
- 保持 main 和 draft model 的状态同步

**is_prefill=true 的作用**:
- 允许 KV cache 裁剪 (crop)
- 更新 position_ids 到正确的位置
- 准备下一轮生成

**对应 Python 实现**:
```python
# dflash/model.py line ~134
output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
start += acceptance_length + 1
past_key_values_target.crop(start)  # <- Main model crop

# line ~119
past_key_values_draft.crop(start)   # <- Draft model crop
```

### Step 7: Cleanup Finished Requests

```cpp
for (const auto& draft_request : draft_generated_requests) {
    auto request_id = draft_request.first;

    if (verified_requests.find(request_id) == verified_requests.end()) {
        m_draft_pipeline->finish_request(request_id);
        m_draft_generations.erase(request_id);
    }

    // Update metrics
    auto updated_seq_info = update_sequence_info[request_id];
    m_sd_metrics.update_draft_generated_len(request_id, updated_seq_info.inserted_tokens_cnt);

    if (updated_seq_info.inserted_tokens_cnt > 0 && !verified_requests.empty()) {
        float acceptance_rate = 1.0f - static_cast<float>(updated_seq_info.removed_tokens_cnt) /
                                       updated_seq_info.inserted_tokens_cnt;
        m_sd_metrics.update_acceptance_rate(request_id, acceptance_rate * 100);
        m_sd_metrics.update_draft_accepted_tokens(request_id,
            updated_seq_info.inserted_tokens_cnt - updated_seq_info.removed_tokens_cnt);
    }
}
```

**目的**:
- 清理已完成的 requests
- 更新性能指标
- 计算 acceptance rate

## 关键设计决策

### 1. 为什么需要两次 Main Model Step？

```
Step 1: Main Model Forward  -> 生成 hidden states
Step 5: Main Model Verify   -> 验证 draft tokens
```

这是 DFlash 算法的核心：
- **第一次**: 生成 context (hidden states) 给 draft model
- **第二次**: 验证 draft tokens，实现推测解码加速

### 2. Position IDs 的正确更新

**Main Model**:
```
Prefill:  [0, 1, 2, ..., seq_len-1]
Generate: [seq_len, seq_len+1, ...]
Verify:   [start, start+1, ..., start+block_size-1]
```

**Draft Model**:
```
Virtual Prefill: [0, 1, 2, ..., current_len-1]
Generate:        [current_len, current_len+1, ..., current_len+block_size-1]
Update:          crop to acceptance_length
```

**关键**: `update_request(..., is_prefill=true)` 会正确处理 position_ids 的更新

### 3. KV Cache 管理

**Paged Attention 中的 KV Cache**:
- 按 block 分配 (例如 block_size=16)
- 需要预先分配足够的 blocks
- 通过 virtual prefill 确保分配正确

**Draft Model 的 KV Cache**:
```
Step 2: Allocate blocks (virtual prefill)
Step 3: Extend cache (multistep generation)
Step 6: Crop cache (remove rejected tokens)
```

## 与 Eagle3 的区别

| 特性 | Eagle3 | DFlash |
|------|--------|--------|
| Draft 输入 | 自己的 embeddings | Main hidden states + embeddings |
| Attention | Causal | Non-causal |
| KV Cache | 独立 | 需要 main hidden states context |
| Prefill | 正常 prefill | Virtual prefill (状态同步) |
| Position IDs | 标准自回归 | 需要精确管理 |

## 调试建议

### 1. 检查 Hidden States 导出

```cpp
// 确认 main model 导出 hidden states
m_main_dflash_pipeline->set_hidden_state_export_needed(true);
```

### 2. 验证 Position IDs

在 update_request 前后打印 position_ids:
```cpp
std::cout << "Before update: position_ids = " << position_ids << std::endl;
auto result = m_draft_pipeline->update_request(...);
std::cout << "After update: position_ids = " << position_ids << std::endl;
```

### 3. 检查 KV Cache Block 分配

```cpp
// Virtual prefill 后检查 blocks
auto blocks_allocated = draft_pipeline->get_allocated_blocks(request_id);
std::cout << "Allocated blocks: " << blocks_allocated << std::endl;
```

### 4. 监控 Acceptance Rate

```cpp
// 应该在 40%-80% 之间
float acceptance_rate = 1.0f - removed / inserted;
OPENVINO_ASSERT(acceptance_rate > 0.2, "Acceptance rate too low!");
```

## 性能优化

1. **Batch Size**: DFlash 支持 continuous batching，可以同时处理多个 requests
2. **Block Size**: 调整 `num_assistant_tokens` 平衡加速比和 acceptance rate
3. **Hidden Layers**: 选择合适的 `hidden_layers` 提供给 draft model

## 参考

- DFlash Paper: https://arxiv.org/abs/2410.13813
- Reference Implementation: https://github.com/z-lab/dflash
- Paged Attention: https://github.com/vllm-project/vllm
