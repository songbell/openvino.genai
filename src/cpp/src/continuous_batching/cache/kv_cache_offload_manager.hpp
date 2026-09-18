// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "continuous_batching/cache/i_kv_cache_storage_backend.hpp"
#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "openvino/genai/cache_offload.hpp"

namespace ov::genai {

/**
 * @brief Fixed-slot disk backend for offloaded KV cache blocks.
 *
 * Owns a run-specific file, hands out slots sized to exactly one block set, and performs
 * length-verified reads and writes. It deliberately knows nothing about sequences or eviction policy -
 * those stay in the block manager and the orchestrator; hashes are used purely as the backend's own
 * lookup key (`m_hash_to_slot`), never for any prefix-cache decision.
 *
 * This is openvino_genai's built-in `DefaultFileStorageBackend` (see `new_plan.md` §3.3.3): the only
 * `IKVCacheStorageBackend` implementation today, and the one `KVCacheOffloadCache` falls back to when no
 * third-party storage plugin is configured.
 */
class KVCacheOffloadManager : public IKVCacheStorageBackend {
public:
    /**
     * @param layout Per-layer key/value byte layout of one block; used for the authoritative slot size
     * and byte offsets. Kept as a constructor parameter (rather than derived solely from the
     * interface's `StorageModelMetadata`, whose `block_bytes_per_layer` assumes uniform layer sizes) so
     * this in-tree backend keeps supporting models with non-uniform per-layer KV byte sizes.
     * @param tenant_isolation_seed Recorded in the persisted manifest header (has no effect when
     * persistence is disabled). A persisted cache is only ever recovered when this matches the seed it
     * was written with, so a cache belonging to a different tenant/cache_salt (see
     * compute_prefix_isolation_seed()) is always rejected and rebuilt fresh rather than partially reused.
     */
    KVCacheOffloadManager(const KVCacheDiskLayout& layout,
                          const CacheOffloadConfig& config,
                          uint64_t tenant_isolation_seed = 0);
    ~KVCacheOffloadManager() override;

    KVCacheOffloadManager(const KVCacheOffloadManager&) = delete;
    KVCacheOffloadManager& operator=(const KVCacheOffloadManager&) = delete;

    /// @return Whether KV cache offload is implemented for this inference device.
    static bool is_supported_device(const std::string& device);

    void initialize(const std::string& storage_path,
                    const StorageModelMetadata& metadata,
                    const ov::AnyMap& custom_properties) override;

    bool has_block(std::uint64_t block_hash) const override;
    std::vector<bool> has_blocks(const std::vector<std::uint64_t>& block_hashes) const override;
    bool write_blocks(const std::vector<BlockIORequest>& requests) override;
    bool read_blocks(const std::vector<BlockIORequest>& requests) override;
    void evict_blocks(const std::vector<std::uint64_t>& block_hashes) override;
    void flush_manifest() override;
    void shutdown() override;

    std::size_t get_num_slots() const override {
        return m_num_slots;
    }

    std::size_t get_num_free_slots() const override;

    /// @return Hashes recovered from a compatible persisted cache at `initialize()` time.
    /// Empty when persistence is disabled, this is the first run for this directory, or the persisted
    /// cache was incompatible (see CacheOffloadConfig::enable_persistence) and was therefore rebuilt fresh.
    const std::vector<std::uint64_t>& get_recovered_hashes() const override {
        return m_recovered_hashes;
    }

    /// @return The offload file's path, for logging/diagnostics only.
    std::string describe() const override {
        return m_file_path.string();
    }

    /// Deletes a persisted cache's files from @p directory, if present. Safe to call on a directory with
    /// no persisted cache (no-op). Intended for explicit cleanup tooling, not called automatically.
    static void remove_persisted_cache(const std::filesystem::path& directory);

private:
    void write_at(int fd, std::size_t offset, const uint8_t* data, std::size_t size) const;
    void read_at(int fd, std::size_t offset, uint8_t* data, std::size_t size) const;
    void close_and_remove() noexcept;

    /// Tries to reuse an existing persisted cache in `m_file_path`/`m_manifest_path`; returns false (and
    /// leaves both files untouched for create_fresh_persisted_files() to overwrite) if either is missing,
    /// unreadable, or incompatible with this run's expected header.
    bool try_recover_persisted_cache(const CacheOffloadConfig& config);
    void create_fresh_persisted_files(const CacheOffloadConfig& config);
    void write_manifest_record(std::size_t slot_id, std::uint64_t hash, std::uint64_t checksum);
    void mark_manifest_invalid(std::size_t slot_id);

    KVCacheDiskLayout m_layout;
    // Byte offset (within one slot) and size of each layer's combined key+value span; index = layer_idx.
    std::vector<std::pair<std::size_t, std::size_t>> m_layer_spans;
    std::filesystem::path m_file_path;
    std::filesystem::path m_manifest_path;
    int m_fd = -1;
    int m_manifest_fd = -1;
    bool m_persistent = false;
    std::size_t m_slot_size = 0;
    std::size_t m_num_slots = 0;
    // 0 means no tenant isolation was configured; recorded in the persisted manifest header and checked
    // on recovery so a cache from a different tenant/cache_salt is never partially reused.
    std::uint64_t m_tenant_isolation_seed = 0;
    std::vector<std::size_t> m_free_slots;
    std::unordered_map<std::uint64_t, std::size_t> m_hash_to_slot;
    // Checksum of each slot's last-written contents, in-memory regardless of persistence, so read_blocks()
    // can always detect corruption; populated by write_blocks() or, for a recovered slot, at initialize().
    std::vector<std::uint64_t> m_slot_checksums;
    std::vector<std::uint64_t> m_recovered_hashes;
    // Serializes slot bookkeeping and the seek/read/write pairs used on platforms without positional I/O.
    mutable std::mutex m_mutex;
};

}  // namespace ov::genai
