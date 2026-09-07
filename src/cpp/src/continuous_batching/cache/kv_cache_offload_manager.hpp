// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
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
 * length-verified reads and writes. It deliberately knows nothing about block hashes,
 * sequences or eviction policy - those stay in the block manager and the orchestrator.
 *
 * This is openvino_genai's built-in `DefaultFileStorageBackend` (see `kv_cache_offload_new_plan.md`
 * Phase 6): the only `IKVCacheStorageBackend` implementation today, and the one
 * `KVCacheOffloadCache` falls back to when no third-party storage plugin is configured.
 */
class KVCacheOffloadManager : public IKVCacheStorageBackend {
public:
    /**
     * @param tenant_isolation_seed Recorded in the persisted manifest header (has no effect when
     * persistence is disabled). A persisted cache is only ever recovered when this matches the seed it
     * was written with, so a cache belonging to a different tenant/cache_salt (see
     * compute_prefix_isolation_seed()) is always rejected and rebuilt fresh rather than partially reused.
     */
    KVCacheOffloadManager(const KVCacheDiskLayout& layout,
                          const CacheOffloadConfig& config,
                          const std::string& device,
                          uint64_t tenant_isolation_seed = 0);
    ~KVCacheOffloadManager() override;

    KVCacheOffloadManager(const KVCacheOffloadManager&) = delete;
    KVCacheOffloadManager& operator=(const KVCacheOffloadManager&) = delete;

    /// @return Whether KV cache offload is implemented for this inference device.
    static bool is_supported_device(const std::string& device);

    /// @return Size of one slot in bytes, equal to the byte size of one block set across all layers.
    std::size_t get_slot_size() const override {
        return m_slot_size;
    }

    /// @return Total number of slots derived from the configured capacity.
    std::size_t get_num_slots() const override {
        return m_num_slots;
    }

    std::size_t get_num_free_slots() const override;

    /// @return A free slot, or std::nullopt when the offload file is full.
    std::optional<std::size_t> acquire_slot() override;

    void release_slot(std::size_t slot_id) override;

    /**
     * @brief Writes one slot's data, computing and recording its integrity checksum.
     * @param hash The prefix-cache hash this slot's contents belong to. Only used to persist the slot
     * index when persistence is enabled (see `CacheOffloadConfig::enable_persistence`); the manager itself
     * never uses it for any eviction or lookup decision. Omit it to leave the slot absent from the
     * persisted index (it will simply not be recoverable after a restart).
     */
    void write_slot(std::size_t slot_id,
                   const std::vector<uint8_t>& block_data,
                   std::optional<std::size_t> hash = std::nullopt) override;

    /**
     * @brief Reads one slot's data and verifies it against the checksum recorded by write_slot() (or, for
     * a slot recovered from a persisted cache, the checksum stored in the on-disk index).
     * @throws ov::Exception if the checksum does not match, so a caller already treating I/O errors as a
     * miss (recomputing instead of trusting corrupted bytes) gets the same safe behavior here.
     */
    void read_slot(std::size_t slot_id, std::vector<uint8_t>& block_data) const override;

    const std::filesystem::path& get_file_path() const {
        return m_file_path;
    }

    /// @return (hash, slot_id) pairs recovered from a compatible persisted cache at construction time.
    /// Empty when persistence is disabled, this is the first run for this directory, or the persisted
    /// cache was incompatible (see CacheOffloadConfig::enable_persistence) and was therefore rebuilt fresh.
    const std::vector<std::pair<std::size_t, std::size_t>>& get_recovered_entries() const override {
        return m_recovered_entries;
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
    void write_manifest_record(std::size_t slot_id, std::size_t hash, std::uint64_t checksum);

    KVCacheDiskLayout m_layout;
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
    // Checksum of each slot's last-written contents, in-memory regardless of persistence, so read_slot()
    // can always detect corruption; populated by write_slot() or, for a recovered slot, at construction.
    std::vector<std::uint64_t> m_slot_checksums;
    std::vector<std::pair<std::size_t, std::size_t>> m_recovered_entries;
    // Serializes slot bookkeeping and the seek/read/write pairs used on platforms without positional I/O.
    mutable std::mutex m_mutex;
};

}  // namespace ov::genai
