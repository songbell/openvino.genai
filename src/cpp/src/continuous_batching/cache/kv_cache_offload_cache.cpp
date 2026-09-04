// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/kv_cache_offload_cache.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

#include "logger.hpp"

namespace ov::genai {

namespace {

// TEMPORARY diagnostics helper, to be removed once the GPU cost breakdown is settled.
class ScopedTimer {
public:
    explicit ScopedTimer(std::size_t& accumulator_us)
        : m_accumulator_us(accumulator_us), m_started(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        m_accumulator_us += static_cast<std::size_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_started).count());
    }

private:
    std::size_t& m_accumulator_us;
    std::chrono::steady_clock::time_point m_started;
};

}  // namespace

KVCacheOffloadCache::KVCacheOffloadCache(KVCacheManager& cache_manager,
                                         std::unique_ptr<KVCacheOffloadManager> backend,
                                                                                 std::size_t max_queued_stores,
                                                                                 bool wait_for_buffer,
                                                                                 bool enable_detailed_logging,
                                                                                 std::size_t host_cache_slots)
    : m_cache_manager(cache_manager),
      m_backend(std::move(backend)),
            m_max_queued_stores(max_queued_stores),
            m_wait_for_buffer(wait_for_buffer),
            m_enable_detailed_logging(enable_detailed_logging),
            m_host_pool(host_cache_slots) {
    OPENVINO_ASSERT(m_backend != nullptr, "KV cache offload backend must not be null");
    OPENVINO_ASSERT(m_max_queued_stores > 0, "KV cache offload needs at least one staging buffer");
    OPENVINO_ASSERT(m_max_queued_stores <= CACHE_OFFLOAD_MAX_BUFFER_SLOTS,
                    "KV cache offload buffer_slots must not exceed ", CACHE_OFFLOAD_MAX_BUFFER_SLOTS);
    OPENVINO_ASSERT(m_backend->get_slot_size() == m_cache_manager.get_block_layout().get_slot_size(),
                    "KV cache offload backend slot size does not match the cache block layout");
    m_writer = std::thread(&KVCacheOffloadCache::run_writer, this);
}

KVCacheOffloadCache::~KVCacheOffloadCache() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stopping = true;
    }
    m_queued_cv.notify_all();
    if (m_writer.joinable()) {
        m_writer.join();
    }
    // TEMPORARY diagnostics, to be removed once the GPU cost breakdown is settled.
    GENAI_INFO("[KV_TRACE] offload_stats stored=%zu replaced=%zu loaded=%zu failed=%zu dropped=%zu "
               "queue_peak=%zu load_staging=%zu load_host=%zu load_disk=%zu load_miss=%zu host_evictions=%zu "
               "store_read=%zu us store_write=%zu us load_disk_time=%zu us load_write=%zu us",
               m_statistics.num_stored,
               m_statistics.num_replaced,
               m_statistics.num_loaded,
               m_statistics.num_failed,
               m_statistics.num_dropped_no_buffer,
               m_statistics.num_queue_peak,
               m_statistics.num_load_staging,
               m_statistics.num_load_host,
               m_statistics.num_load_disk,
               m_statistics.num_load_misses,
               m_statistics.num_host_evictions,
               m_statistics.store_read_us,
               m_statistics.store_write_us,
               m_statistics.load_disk_us,
               m_statistics.load_write_us);
}

void KVCacheOffloadCache::on_blocks_overwritten(std::size_t hash, const BlocksPerLayer& blocks) {
    // Offload is only enabled without cache eviction, where one block table is shared by all decoder
    // layers and a single physical block index therefore addresses the whole block set.
    OPENVINO_ASSERT(blocks.size() == 1,
                    "KV cache offload requires the shared block table used when cache eviction is disabled, got ",
                    blocks.size(),
                    " block-table layers");
    const auto block_index = blocks.front()->get_index();
    OPENVINO_ASSERT(block_index >= 0, "Invalid physical block index ", block_index);

    std::unique_lock<std::mutex> lock(m_mutex);

    if (locate_unlocked(hash).is_known()) {
        // Contents are content-addressed by `hash`, so the stored copy already holds the same data.
        ++m_statistics.num_already_present;
        return;
    }

    if (m_wait_for_buffer) {
        m_drained_cv.wait(lock, [this] {
            return m_stopping || m_queued_stores.size() < m_max_queued_stores;
        });
        if (m_stopping) {
            ++m_statistics.num_failed;
            return;
        }
    } else if (m_queued_stores.size() >= m_max_queued_stores) {
        ++m_statistics.num_dropped_no_buffer;
        return;
    }

    std::optional<std::size_t> slot_id = m_backend->acquire_slot();
    bool reclaimed = false;
    if (!slot_id.has_value()) {
        if (m_reclamation_paused) {
            ++m_statistics.num_failed;
            return;
        }
        slot_id = reclaim_oldest_slot();
        reclaimed = slot_id.has_value();
    }
    if (!slot_id.has_value()) {
        ++m_statistics.num_failed;
        return;
    }

    QueuedStore queued;
    queued.hash = hash;
    queued.slot_id = *slot_id;
    queued.reclaimed = reclaimed;
    try {
        // Copy the contents out now; the block is handed over for overwriting as soon as this returns.
        ScopedTimer timer(m_statistics.store_read_us);
        queued.data = m_cache_manager.read_block(static_cast<std::size_t>(block_index));
    } catch (const std::exception& error) {
        GENAI_WARN("KV cache offload store failed for hash %zu: %s", hash, error.what());
        m_backend->release_slot(*slot_id);
        ++m_statistics.num_failed;
        return;
    }

    m_queued_stores.push_back(std::move(queued));
    m_statistics.num_queue_peak = std::max(m_statistics.num_queue_peak, m_queued_stores.size());
    // L0 -> L1 happens now, independent of whether the disk write below ever completes: a hash becomes
    // host-visible as soon as it leaves the device, not only after a durable disk round-trip. `queued.data`
    // was moved above, so this reads the buffered copy back out of the deque entry that now owns it.
    if (m_host_pool.put(hash, m_queued_stores.back().data).has_value()) {
        ++m_statistics.num_host_evictions;
    }
    if (m_enable_detailed_logging) {
        GENAI_INFO("[KV_TRACE] KVCacheOffloadCache queue_store hash=%zu physical_block=%d queued=%zu",
                   hash,
                   block_index,
                   m_queued_stores.size());
    }
    m_queued_cv.notify_one();
}

void KVCacheOffloadCache::run_writer() {
    std::unique_lock<std::mutex> lock(m_mutex);
    while (true) {
        m_queued_cv.wait(lock, [this] { return m_stopping || !m_queued_stores.empty(); });
        if (m_queued_stores.empty()) {
            if (m_stopping) {
                return;
            }
            continue;
        }

        // Only this thread pops, and deque never invalidates references to existing elements,
        // so the front entry stays readable while the mutex is released for the write.
        const QueuedStore& front = m_queued_stores.front();
        const std::size_t hash = front.hash;
        const std::size_t slot_id = front.slot_id;
        const bool reclaimed = front.reclaimed;

        bool stored = true;
        lock.unlock();
        const auto started = std::chrono::steady_clock::now();
        try {
            m_backend->write_slot(slot_id, front.data);
        } catch (const std::exception& error) {
            GENAI_WARN("KV cache offload store failed for hash %zu: %s", hash, error.what());
            stored = false;
        }
        const auto elapsed_us = static_cast<std::size_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count());
        lock.lock();
        m_statistics.store_write_us += elapsed_us;

        m_queued_stores.pop_front();
        if (stored) {
            publish(hash, slot_id, reclaimed);
        } else {
            m_backend->release_slot(slot_id);
            ++m_statistics.num_failed;
        }
        m_drained_cv.notify_all();
    }
}

void KVCacheOffloadCache::publish(std::size_t hash, std::size_t slot_id, bool reclaimed) {
    m_insertion_order.push_back(hash);
    m_entries[hash] = Entry{slot_id, std::prev(m_insertion_order.end())};
    ++m_statistics.num_stored;
    if (reclaimed) {
        ++m_statistics.num_replaced;
    }
    // The L1 host copy was already seeded in on_blocks_overwritten; this only records disk durability and
    // must not re-touch or re-evict the independent host tier.
    if (m_enable_detailed_logging) {
        GENAI_INFO("[KV_TRACE] KVCacheOffloadCache publish hash=%zu slot=%zu entries=%zu replaced=%s",
                   hash,
                   slot_id,
                   m_entries.size(),
                   reclaimed ? "true" : "false");
    }
}

void KVCacheOffloadCache::flush() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_drained_cv.wait(lock, [this] { return m_queued_stores.empty(); });
}

const KVCacheOffloadCache::QueuedStore* KVCacheOffloadCache::find_queued(std::size_t hash) const {
    for (const auto& queued : m_queued_stores) {
        if (queued.hash == hash) {
            return &queued;
        }
    }
    return nullptr;
}

KVCacheOffloadCache::BlockLocation KVCacheOffloadCache::locate_unlocked(std::size_t hash) const {
    BlockLocation location;
    location.in_flight = find_queued(hash) != nullptr;
    location.host_resident = m_host_pool.contains(hash);
    location.disk_resident = m_entries.find(hash) != m_entries.end();
    return location;
}

KVCacheOffloadCache::BlockLocation KVCacheOffloadCache::get_location(std::size_t hash) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return locate_unlocked(hash);
}

KVCacheOffloadCache::ScopedReclamationPause::ScopedReclamationPause(KVCacheOffloadCache& cache) : m_cache(cache) {
    std::lock_guard<std::mutex> lock(m_cache.m_mutex);
    m_cache.m_reclamation_paused = true;
}

KVCacheOffloadCache::ScopedReclamationPause::~ScopedReclamationPause() {
    std::lock_guard<std::mutex> lock(m_cache.m_mutex);
    m_cache.m_reclamation_paused = false;
}

std::optional<std::size_t> KVCacheOffloadCache::reclaim_oldest_slot() {
    if (m_insertion_order.empty()) {
        return std::nullopt;
    }
    const std::size_t oldest_hash = m_insertion_order.front();
    auto it = m_entries.find(oldest_hash);
    OPENVINO_ASSERT(it != m_entries.end(), "KV cache offload index and insertion order are out of sync");
    const std::size_t slot_id = it->second.slot_id;
    m_insertion_order.pop_front();
    m_entries.erase(it);
    return slot_id;
}

bool KVCacheOffloadCache::contains(std::size_t hash) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return locate_unlocked(hash).is_known();
}

bool KVCacheOffloadCache::load_into(std::size_t hash, std::size_t block_index) {
    std::lock_guard<std::mutex> lock(m_mutex);

    // Branch order mirrors BlockLocation's priority: in_flight > host_resident > disk_resident.
    try {
        if (const QueuedStore* queued = find_queued(hash)) {
            // Still on its way to disk, so the staging copy is the closest source.
            ++m_statistics.num_load_staging;
            if (m_enable_detailed_logging) {
                GENAI_INFO("[KV_TRACE] KVCacheOffloadCache load hash=%zu source=staging physical_block=%zu",
                           hash,
                           block_index);
            }
            ScopedTimer timer(m_statistics.load_write_us);
            m_cache_manager.write_block(block_index, queued->data);
        } else if (const std::vector<uint8_t>* host_data = m_host_pool.get(hash)) {
            // L1 hit: skip the disk entirely.
            ++m_statistics.num_load_host;
            if (m_enable_detailed_logging) {
                GENAI_INFO("[KV_TRACE] KVCacheOffloadCache load hash=%zu source=host physical_block=%zu",
                           hash,
                           block_index);
            }
            ScopedTimer timer(m_statistics.load_write_us);
            m_cache_manager.write_block(block_index, *host_data);
        } else {
            auto it = m_entries.find(hash);
            if (it == m_entries.end()) {
                ++m_statistics.num_load_misses;
                return false;
            }
            ++m_statistics.num_load_disk;
            if (m_enable_detailed_logging) {
                GENAI_INFO("[KV_TRACE] KVCacheOffloadCache load hash=%zu source=disk slot=%zu physical_block=%zu",
                           hash,
                           it->second.slot_id,
                           block_index);
            }
            {
                ScopedTimer timer(m_statistics.load_disk_us);
                m_backend->read_slot(it->second.slot_id, m_staging);
            }
            ScopedTimer timer(m_statistics.load_write_us);
            m_cache_manager.write_block(block_index, m_staging);
        }
    } catch (const std::exception& error) {
        GENAI_WARN("KV cache offload load failed for hash %zu: %s", hash, error.what());
        ++m_statistics.num_failed;
        return false;
    }

    ++m_statistics.num_loaded;
    if (m_enable_detailed_logging) {
        GENAI_INFO("[KV_TRACE] offload_load_block index=%zu", block_index);
    }
    return true;
}

bool KVCacheOffloadCache::read(std::size_t hash, std::vector<uint8_t>& block_data) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (const QueuedStore* queued = find_queued(hash)) {
        block_data = queued->data;
        return true;
    }
    if (const std::vector<uint8_t>* host_data = m_host_pool.get(hash)) {
        block_data = *host_data;
        return true;
    }
    auto it = m_entries.find(hash);
    if (it == m_entries.end()) {
        return false;
    }
    m_backend->read_slot(it->second.slot_id, block_data);
    return true;
}

std::size_t KVCacheOffloadCache::get_num_entries() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_entries.size();
}

std::size_t KVCacheOffloadCache::get_num_host_entries() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_host_pool.size();
}

KVCacheOffloadCache::Statistics KVCacheOffloadCache::get_statistics() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_statistics;
}

}  // namespace ov::genai
