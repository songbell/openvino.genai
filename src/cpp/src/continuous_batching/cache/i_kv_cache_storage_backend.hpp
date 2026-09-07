// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/**
 * @brief Caller-supplied identity of the model/tokenizer/KV layout a storage backend is asked to hold
 * blocks for (`new_plan.md` §3.3.1). A backend uses this to validate a persisted cache against the
 * current run before trusting any of it, and never to make prefix-cache policy decisions.
 */
struct StorageModelMetadata {
    std::string model_name;
    std::string model_fingerprint;  // model structure/weights identity
    std::string tokenizer_hash;
    ov::element::Type kv_precision;
    std::size_t kv_block_size_tokens = 0;
    std::size_t num_layers = 0;
    std::size_t block_bytes_per_layer = 0;  // combined key+value bytes for one layer of one block
};

/**
 * @brief Describes one layer's worth of I/O for one block.
 *
 * A whole block's write/read is expressed as `num_layers` of these sharing the same `block_hash`,
 * batched together in a single call to `write_blocks()`/`read_blocks()` - never split across calls - so
 * a backend can validate and checksum a whole block's worth of data at once.
 *
 * `host_buffer` is owned by the caller (`KVCacheOffloadCache`): a backend must not free it, must not
 * retain the pointer past a synchronous call's return (or past the completion callback of an
 * asynchronous one), and must never access outside `buffer_size_bytes`.
 */
struct BlockIORequest {
    std::uint64_t block_hash = 0;
    void* host_buffer = nullptr;
    std::size_t buffer_size_bytes = 0;
    std::uint32_t layer_idx = 0;
};

using AsyncIOCompletionCallback = std::function<void(int status_code, std::size_t completed_blocks)>;

/**
 * @brief Boundary between the KV cache offload cache (hash -> slot index, publish/load orchestration,
 * eviction policy) and a slot storage backend (hash -> bytes).
 *
 * A backend owns its own hash -> internal-slot mapping and moves per-layer byte buffers in and out; it
 * is deliberately kept ignorant of block hashes' *meaning*, sequences, `RemoteTensor`, or any
 * prefix-cache policy - all of that stays in `KVCacheOffloadCache`/`BlockManager`. This is the seam a
 * third-party storage plugin would sit behind (see `new_plan.md` §3.3): `KVCacheOffloadCache` only ever
 * talks to this interface, never to a concrete backend type.
 *
 * `KVCacheOffloadManager` (the in-tree `DefaultFileStorageBackend`) is the only implementation today.
 */
class IKVCacheStorageBackend {
public:
    virtual ~IKVCacheStorageBackend() = default;

    /// Initializes the backend: mounts/creates the storage at @p storage_path, and validates or rebuilds
    /// its metadata against @p metadata. Called exactly once, before any other method.
    virtual void initialize(const std::string& storage_path,
                            const StorageModelMetadata& metadata,
                            const ov::AnyMap& custom_properties) = 0;

    /// @return Whether @p block_hash's contents are currently held by this backend.
    virtual bool has_block(std::uint64_t block_hash) const = 0;

    /// @return Per-hash presence, in the same order as @p block_hashes.
    virtual std::vector<bool> has_blocks(const std::vector<std::uint64_t>& block_hashes) const = 0;

    /// @brief Writes the given per-layer requests, which may span multiple blocks. Requests that share a
    /// `block_hash` must all be present in this same call (never split across calls), so the backend can
    /// validate and checksum a whole block's worth of data at once.
    /// @return Whether every request was written successfully.
    virtual bool write_blocks(const std::vector<BlockIORequest>& requests) = 0;

    /// @brief Reads the given per-layer requests into their `host_buffer`s. Same whole-block-per-call
    /// requirement as `write_blocks()`.
    /// @return Whether every request was read successfully and passed its integrity check.
    virtual bool read_blocks(const std::vector<BlockIORequest>& requests) = 0;

    /// @brief Asynchronous variant for prefetch pipelines. Default implementation is a synchronous
    /// `read_blocks()` call followed immediately by @p callback; a backend with real async I/O overrides
    /// this and calls @p callback from whichever thread completes the operation.
    virtual void async_read_blocks(const std::vector<BlockIORequest>& requests,
                                   const AsyncIOCompletionCallback& callback) {
        const bool ok = read_blocks(requests);
        callback(ok ? 0 : -1, ok ? requests.size() : 0);
    }

    /// Evicts the given hashes, freeing their storage for reuse. The caller (`KVCacheOffloadCache`) alone
    /// decides which hashes to evict and when (LRU); a backend must never evict on its own initiative.
    virtual void evict_blocks(const std::vector<std::uint64_t>& block_hashes) = 0;

    /// Flushes any buffered metadata/index state to durable storage.
    virtual void flush_manifest() = 0;

    /// Releases storage resources. Called once, after which no other method is called.
    virtual void shutdown() = 0;

    /// @return Total number of blocks this backend can hold at once.
    virtual std::size_t get_num_slots() const = 0;

    /// @return Number of additional blocks that can be written before the backend is full.
    virtual std::size_t get_num_free_slots() const = 0;

    /// @return Hashes recovered from a compatible persisted cache at `initialize()` time. Always empty
    /// for a backend that does not support persistence, or on the first run for a given storage path.
    virtual const std::vector<std::uint64_t>& get_recovered_hashes() const = 0;

    /// @return A short human-readable description (e.g. a file path) for logging/diagnostics only; never
    /// parsed or relied upon for behavior.
    virtual std::string describe() const = 0;
};

}  // namespace ov::genai
