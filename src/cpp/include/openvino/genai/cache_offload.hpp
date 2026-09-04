// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <sstream>
#include <string>

namespace ov::genai {

constexpr std::size_t CACHE_OFFLOAD_MAX_BUFFER_SLOTS = 1024;

/**
 * @brief Configuration of the KV cache disk offload backend.
 *
 * Offloaded blocks are rediscovered through the prefix-cache block hash, so offload is only
 * meaningful together with `SchedulerConfig::enable_prefix_caching`.
 */
struct CacheOffloadConfig {
    /** Existing directory to place the offload file in. When empty, the system temporary directory is used. */
    std::string path;

    /** Upper bound of the offload file size in bytes. The usable slot count is derived from this. */
    std::size_t capacity_bytes = 0;

    /** Number of staging buffers reserved for transfers; each buffer holds one complete block snapshot. */
    std::size_t buffer_slots = 2;

    /** Whether store producers wait for a staging buffer instead of dropping a store when the queue is full. */
    bool wait_for_buffer = false;

    /** Whether per-block offload trace messages are enabled. */
    bool enable_detailed_logging = false;

    /** Whether the offload file may go through the OS page cache. Direct I/O is not implemented yet. */
    bool use_page_cache = true;

    /**
     * Number of block snapshots kept in an independent host-memory (L1) cache after their disk write
     * completes, so repeated hits can skip the disk read. 0 disables this tier; entries are dropped from
     * host memory as soon as they reach disk, matching the pre-Phase-2 behavior.
     */
    std::size_t host_cache_slots = 0;

    bool operator==(const CacheOffloadConfig& other) const {
        return path == other.path && capacity_bytes == other.capacity_bytes &&
             buffer_slots == other.buffer_slots && wait_for_buffer == other.wait_for_buffer &&
             enable_detailed_logging == other.enable_detailed_logging && use_page_cache == other.use_page_cache &&
             host_cache_slots == other.host_cache_slots;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "  CacheOffloadConfig { \n";
        oss << "    path: " << (path.empty() ? std::string("<temporary directory>") : path) << "\n";
        oss << "    capacity_bytes: " << capacity_bytes << "\n";
        oss << "    buffer_slots: " << buffer_slots << "\n";
        oss << "    wait_for_buffer: " << std::boolalpha << wait_for_buffer << "\n";
        oss << "    enable_detailed_logging: " << std::boolalpha << enable_detailed_logging << "\n";
        oss << "    use_page_cache: " << std::boolalpha << use_page_cache << "\n";
        oss << "    host_cache_slots: " << host_cache_slots << "\n";
        oss << "  }";
        return oss.str();
    }
};

}  // namespace ov::genai
