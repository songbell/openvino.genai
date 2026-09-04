// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <unordered_map>
#include <vector>

namespace ov::genai {

/**
 * @brief Fixed-capacity host-memory cache of KV block byte snapshots (the L1 tier in the
 * L0 device -> L1 host -> L2 disk offload hierarchy).
 *
 * Independent from the disk backend and from the transient in-flight store queue in
 * `KVCacheOffloadCache`: an entry placed here persists after the corresponding disk write
 * completes, so repeated hits can be served without touching the backing file. Eviction is LRU
 * and only ever drops the host-resident copy - it never invalidates a hash's disk-backed entry,
 * since the two tiers are tracked independently.
 */
class HostBlockPool {
public:
    explicit HostBlockPool(std::size_t capacity) : m_capacity(capacity) {}

    /**
     * @brief Inserts or refreshes the host copy for @p hash, evicting the least recently used
     * entry if the pool is at capacity. A capacity of zero disables the pool: nothing is stored.
     * @return The hash evicted to make room, or std::nullopt if nothing was evicted.
     */
    std::optional<std::size_t> put(std::size_t hash, std::vector<uint8_t> data) {
        if (m_capacity == 0) {
            return std::nullopt;
        }

        auto existing = m_entries.find(hash);
        if (existing != m_entries.end()) {
            m_order.splice(m_order.end(), m_order, existing->second.order_it);
            existing->second.data = std::move(data);
            return std::nullopt;
        }

        std::optional<std::size_t> evicted;
        if (m_entries.size() >= m_capacity) {
            const std::size_t oldest = m_order.front();
            m_order.pop_front();
            m_entries.erase(oldest);
            evicted = oldest;
        }

        m_order.push_back(hash);
        m_entries.emplace(hash, Entry{std::prev(m_order.end()), std::move(data)});
        return evicted;
    }

    /// @return The cached bytes for @p hash, refreshing its recency, or nullptr if not resident.
    const std::vector<uint8_t>* get(std::size_t hash) {
        auto it = m_entries.find(hash);
        if (it == m_entries.end()) {
            return nullptr;
        }
        m_order.splice(m_order.end(), m_order, it->second.order_it);
        return &it->second.data;
    }

    bool contains(std::size_t hash) const {
        return m_entries.find(hash) != m_entries.end();
    }

    /// Drops the host-resident copy for @p hash, if any. Does not affect any disk-backed entry.
    void erase(std::size_t hash) {
        auto it = m_entries.find(hash);
        if (it == m_entries.end()) {
            return;
        }
        m_order.erase(it->second.order_it);
        m_entries.erase(it);
    }

    std::size_t size() const {
        return m_entries.size();
    }

    std::size_t capacity() const {
        return m_capacity;
    }

    void clear() {
        m_entries.clear();
        m_order.clear();
    }

private:
    struct Entry {
        std::list<std::size_t>::iterator order_it;
        std::vector<uint8_t> data;
    };

    std::size_t m_capacity;
    std::unordered_map<std::size_t, Entry> m_entries;
    // Front is the least recently used hash and the first candidate for eviction.
    std::list<std::size_t> m_order;
};

}  // namespace ov::genai
