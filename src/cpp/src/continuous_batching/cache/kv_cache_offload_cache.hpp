// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/i_kv_cache_storage_backend.hpp"
#include "continuous_batching/cache/kv_cache_host_block_pool.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_offload_manager.hpp"

namespace ov::genai {

/**
 * @brief Second-level prefix cache holding KV block contents on disk.
 *
 * Blocks are written when the in-memory prefix cache is about to overwrite them, which is the last
 * point where their contents are still intact. The contents are copied out synchronously at that
 * moment and reach the file on a background thread, so the allocation path does not wait for disk.
 * A queued block is served straight from its staging copy, which makes the hand-off invisible to
 * callers. Entries are keyed by the prefix-cache block hash; once the backing file is full the
 * oldest entry is replaced.
 *
 * Offload is best effort: a failed store is reported through the statistics and never propagates to
 * the caller, since losing a cache entry only costs recomputation.
 *
 * This is `new_plan.md`'s `TieredCacheManager` (see its architecture diagram, §2): `m_entries`/
 * `m_insertion_order` are the L0/L1/L2 migration-state tracking and hit/miss decision engine
 * (`get_location()`/`resolve_unlocked()`), `m_queued_stores`/the writer thread are the async
 * swap-out queue, and `m_host_pool` (`HostBlockPool`, the L1 tier) is consulted before falling
 * through to the L2 disk backend (`m_backend`, an `IKVCacheStorageBackend`).
 */
class KVCacheOffloadCache : public IOverwrittenBlockObserver, public IExternalPrefixSource {
public:
    struct Statistics {
        std::size_t num_stored = 0;
        std::size_t num_replaced = 0;
        std::size_t num_already_present = 0;
        std::size_t num_loaded = 0;
        std::size_t num_failed = 0;
        std::size_t num_dropped_no_buffer = 0;
        std::size_t num_queue_peak = 0;
        std::size_t num_load_staging = 0;
        std::size_t num_load_disk = 0;
        std::size_t num_load_host = 0;
        std::size_t num_load_misses = 0;
        std::size_t num_host_evictions = 0;
        // TEMPORARY diagnostics, to be removed once the GPU cost breakdown is settled.
        std::size_t store_read_us = 0;
        std::size_t store_write_us = 0;
        std::size_t load_disk_us = 0;
        std::size_t load_write_us = 0;
    };

    /**
     * @brief Keeps existing entries from being replaced for as long as it exists.
     *
     * Warming a prefix chain evicts memory blocks in order to host that very chain, so persisting them
     * must not be allowed to reclaim a slot the chain still has to read back.
     */
    class ScopedReclamationPause {
    public:
        explicit ScopedReclamationPause(KVCacheOffloadCache& cache);
        ~ScopedReclamationPause();

        ScopedReclamationPause(const ScopedReclamationPause&) = delete;
        ScopedReclamationPause& operator=(const ScopedReclamationPause&) = delete;

    private:
        KVCacheOffloadCache& m_cache;
    };

    /**
     * @brief Where a hash's contents currently live. `DEVICE` residency is intentionally not modeled here:
     * it is owned by `BlockManager`/`OverwritableBlocksHashStore`, which this class only ever observes
     * through `on_blocks_overwritten()`, and mirroring it here would create a second, driftable source of
     * truth. `HOST` and `DISK` are independent tiers with their own eviction, so both bits can be set at
     * once; `in_flight` and `host_resident` can likewise both be true, since the L1 host copy is seeded as
     * soon as the bytes leave the device, before the disk write below has necessarily completed.
     */
    struct BlockLocation {
        bool in_flight = false;      // read from device, durable disk write not yet complete
        bool host_resident = false;  // byte copy present in the L1 host pool
        bool disk_resident = false;  // durable copy present in the L2 backing file

        bool is_known() const {
            return in_flight || host_resident || disk_resident;
        }
    };

    KVCacheOffloadCache(KVCacheManager& cache_manager,
                        std::unique_ptr<IKVCacheStorageBackend> backend,
                        std::size_t max_queued_stores = 2,
                        bool wait_for_buffer = false,
                        bool enable_detailed_logging = false,
                        std::size_t host_cache_slots = 0);
    ~KVCacheOffloadCache();

    void on_blocks_overwritten(std::size_t hash, const BlocksPerLayer& blocks) override;

    bool contains(std::size_t hash) const override;

    bool load_into(std::size_t hash, std::size_t block_index) override;

    /// Overridden to hold the lock once for the whole chain instead of once per request.
    std::vector<bool> load_into_many(const std::vector<std::pair<std::size_t, std::size_t>>& requests) override;

    /// @return Which tier(s), if any, currently hold @p hash's contents.
    BlockLocation get_location(std::size_t hash) const;

    /// Waits until every queued store has reached the backing file.
    void flush();

    /**
     * @brief Reads the stored contents for @p hash.
     * @return false if @p hash has no entry, in which case @p block_data is left untouched.
     */
    bool read(std::size_t hash, std::vector<uint8_t>& block_data) const;

    std::size_t get_num_entries() const;

    /// @return Number of block snapshots currently resident in the L1 host cache.
    std::size_t get_num_host_entries() const;

    std::size_t get_num_free_slots() const {
        return m_backend->get_num_free_slots();
    }

    Statistics get_statistics() const;

private:
    struct Entry {
        std::size_t slot_id = 0;
        std::list<std::size_t>::iterator order_it;
    };

    /// A block whose contents are already copied out but not yet on disk.
    struct QueuedStore {
        std::size_t hash = 0;
        std::size_t slot_id = 0;
        bool reclaimed = false;
        std::vector<uint8_t> data;
    };

    /// @return A slot freed by dropping the oldest entry, or std::nullopt when there is nothing to drop.
    std::optional<std::size_t> reclaim_oldest_slot();

    const QueuedStore* find_queued(std::size_t hash) const;

    /// @return The unified location for @p hash. Caller must already hold `m_mutex`.
    BlockLocation locate_unlocked(std::size_t hash) const;

    /// @return Whether @p block_index was filled for @p hash. Caller must already hold `m_mutex`.
    bool load_into_unlocked(std::size_t hash, std::size_t block_index);

    /// @return Whether source bytes for @p hash were resolved into @p destination, without writing to the
    /// device. Caller must already hold `m_mutex`.
    bool resolve_unlocked(std::size_t hash, std::vector<uint8_t>& destination);

    void publish(std::size_t hash, std::size_t slot_id, bool reclaimed);

    void run_writer();

    KVCacheManager& m_cache_manager;
    std::unique_ptr<IKVCacheStorageBackend> m_backend;
    std::unordered_map<std::size_t, Entry> m_entries;
    // Front is the oldest entry and the first to be replaced when the file is full.
    std::list<std::size_t> m_insertion_order;
    // A store stays queued until it is published, so readers never lose sight of it.
    std::deque<QueuedStore> m_queued_stores;
    std::size_t m_max_queued_stores;
    bool m_wait_for_buffer;
    bool m_enable_detailed_logging;
    // Independent L1 tier: entries here outlive their disk write and are evicted by their own LRU.
    // Mutable because read-only accessors (e.g. `read()`) still need to refresh LRU recency on a hit.
    mutable HostBlockPool m_host_pool;
    std::vector<uint8_t> m_staging;
    Statistics m_statistics;
    bool m_reclamation_paused = false;
    bool m_stopping = false;
    mutable std::mutex m_mutex;
    std::condition_variable m_queued_cv;
    std::condition_variable m_drained_cv;
    std::thread m_writer;
};

}  // namespace ov::genai
