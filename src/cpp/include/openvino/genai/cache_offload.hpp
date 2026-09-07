// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <sstream>
#include <string>

#include "openvino/core/any.hpp"

namespace ov::genai {

constexpr std::size_t CACHE_OFFLOAD_MAX_BUFFER_SLOTS = 1024;

/**
 * @brief Configuration of the KV cache disk offload backend.
 *
 * Offloaded blocks are rediscovered through the prefix-cache block hash, so offload is only
 * meaningful together with `SchedulerConfig::enable_prefix_caching`.
 *
 * Field names follow `new_plan.md` §5.1 wherever that document names a field with the same meaning and
 * unit as what already existed here (`storage_cache_dir`, `storage_backend_type`, `storage_plugin_path`,
 * `storage_plugin_properties`). Two fields are deliberately NOT renamed to their §5.1 counterpart:
 * - `host_cache_slots` stays a slot *count* (what `HostBlockPool` actually takes); §5.1's
 *   `host_cache_size` is a GB *budget*, a different unit that would need a byte-size-dependent
 *   conversion this struct cannot do on its own (the byte size of a block is only known once a model
 *   is loaded). Renaming the field without also changing its unit would be misleading.
 * - `capacity_bytes` stays byte-precise and is what the implementation actually uses; §5.1's
 *   `storage_cache_size` is added below as an optional GB-granularity convenience on top of it (mirrors
 *   `SchedulerConfig::cache_size` vs `num_kv_blocks`), not a replacement - tests and small models need
 *   byte, not gigabyte, precision.
 */
struct CacheOffloadConfig {
    /** Existing directory to place the offload file in. When empty, the system temporary directory is used. */
    std::string storage_cache_dir;

    /** Upper bound of the offload file size in bytes. The usable slot count is derived from this. */
    std::size_t capacity_bytes = 0;

    /**
     * Upper bound of the offload file size in GiB, as a coarser alternative to `capacity_bytes`. Only
     * used when greater than 0 and `capacity_bytes` is 0, in which case `capacity_bytes` is derived as
     * `storage_cache_size * 1024^3`.
     */
    std::size_t storage_cache_size = 0;

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

    /**
     * Whether the offload file and its slot index survive process exit, to be reused by a later run
     * instead of being deleted when this run ends. Requires a non-empty `storage_cache_dir` (an
     * explicit, stable directory - the system temporary directory is not appropriate for data meant to
     * outlive the run) and non-empty `model_fingerprint` / `tokenizer_fingerprint`.
     */
    bool enable_persistence = false;

    /**
     * Caller-supplied identity for the model (including precision/shape choices that affect the KV cache
     * layout) backing this cache. Only used when `enable_persistence` is true: an existing persisted cache
     * is reused only when this matches what was stored with it, so a leftover cache from a different model
     * is never misread as valid for the current one.
     */
    std::string model_fingerprint;

    /** Caller-supplied identity for the tokenizer, checked the same way as `model_fingerprint`. */
    std::string tokenizer_fingerprint;

    /**
     * Selects the storage backend implementation: `"default"` (the built-in file-based backend) is the
     * only supported value today. A non-"default" value throws at pipeline construction time - see
     * `new_plan.md` Phase 4, not yet implemented - rather than silently falling back to "default".
     */
    std::string storage_backend_type = "default";

    /**
     * Path to a third-party storage plugin shared library to load in place of the default backend.
     * Not yet implemented (`new_plan.md` Phase 4): a non-empty value throws at pipeline construction
     * time rather than being silently ignored.
     */
    std::string storage_plugin_path;

    /**
     * Vendor-specific properties passed through to a loaded storage plugin. Not yet implemented
     * (`new_plan.md` Phase 4): a non-empty map throws at pipeline construction time rather than being
     * silently ignored.
     */
    ov::AnyMap storage_plugin_properties;

    bool operator==(const CacheOffloadConfig& other) const {
        return storage_cache_dir == other.storage_cache_dir && capacity_bytes == other.capacity_bytes &&
             storage_cache_size == other.storage_cache_size &&
             buffer_slots == other.buffer_slots && wait_for_buffer == other.wait_for_buffer &&
             enable_detailed_logging == other.enable_detailed_logging && use_page_cache == other.use_page_cache &&
             host_cache_slots == other.host_cache_slots && enable_persistence == other.enable_persistence &&
             model_fingerprint == other.model_fingerprint && tokenizer_fingerprint == other.tokenizer_fingerprint &&
             storage_backend_type == other.storage_backend_type && storage_plugin_path == other.storage_plugin_path;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "  CacheOffloadConfig { \n";
        oss << "    storage_cache_dir: " << (storage_cache_dir.empty() ? std::string("<temporary directory>") : storage_cache_dir) << "\n";
        oss << "    capacity_bytes: " << capacity_bytes << "\n";
        oss << "    storage_cache_size: " << storage_cache_size << " GiB\n";
        oss << "    buffer_slots: " << buffer_slots << "\n";
        oss << "    wait_for_buffer: " << std::boolalpha << wait_for_buffer << "\n";
        oss << "    enable_detailed_logging: " << std::boolalpha << enable_detailed_logging << "\n";
        oss << "    use_page_cache: " << std::boolalpha << use_page_cache << "\n";
        oss << "    host_cache_slots: " << host_cache_slots << "\n";
        oss << "    enable_persistence: " << std::boolalpha << enable_persistence << "\n";
        oss << "    model_fingerprint: " << model_fingerprint << "\n";
        oss << "    tokenizer_fingerprint: " << tokenizer_fingerprint << "\n";
        oss << "    storage_backend_type: " << storage_backend_type << "\n";
        oss << "    storage_plugin_path: " << storage_plugin_path << "\n";
        oss << "  }";
        return oss.str();
    }
};

}  // namespace ov::genai
