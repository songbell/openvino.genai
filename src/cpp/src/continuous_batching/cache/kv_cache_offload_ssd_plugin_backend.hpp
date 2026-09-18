// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "continuous_batching/cache/i_kv_cache_storage_backend.hpp"
#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "openvino/genai/c/ssd_plugin_interface.h"
#include "openvino/genai/cache_offload.hpp"

namespace ov::genai {

/**
 * @brief `IKVCacheStorageBackend` adapter that loads a third-party vendor SSD plugin shared library at
 * runtime and delegates every call across the C ABI defined in `openvino/genai/c/ssd_plugin_interface.h`
 * (see `new_plan.md` SS3.3/Phase 4).
 *
 * Construction dlopen's/LoadLibrary's `config.storage_plugin_path`, resolves and validates the
 * `ov_genai_get_ssd_plugin` entry point, and calls the plugin's `init()` - all synchronously, and all
 * failures (library not found, missing symbol, API version mismatch, `init()` returning non-zero) throw
 * `ov::Exception` rather than falling back to the default backend: a plugin the caller explicitly asked
 * for that cannot be loaded is a configuration error to surface, not silently paper over (`new_plan.md`
 * SS7 risk #2).
 *
 * Requires every decoder layer to have the same combined key+value byte size, since
 * `ov_genai_storage_metadata_t::block_bytes_per_layer` (unlike the in-tree
 * `DefaultFileStorageBackend`/`KVCacheOffloadManager`, which is handed the full per-layer
 * `KVCacheDiskLayout`) has no way to express non-uniform per-layer sizes over the C ABI.
 */
class KVCacheOffloadSSDPluginBackend : public IKVCacheStorageBackend {
public:
    KVCacheOffloadSSDPluginBackend(const KVCacheDiskLayout& layout, const CacheOffloadConfig& config);
    ~KVCacheOffloadSSDPluginBackend() override;

    KVCacheOffloadSSDPluginBackend(const KVCacheOffloadSSDPluginBackend&) = delete;
    KVCacheOffloadSSDPluginBackend& operator=(const KVCacheOffloadSSDPluginBackend&) = delete;

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

    std::size_t get_num_slots() const override;
    std::size_t get_num_free_slots() const override;
    const std::vector<std::uint64_t>& get_recovered_hashes() const override {
        return m_recovered_hashes;
    }
    std::string describe() const override;

private:
    void load_library(const std::string& plugin_path);
    void unload_library() noexcept;

    std::string m_plugin_path;
    void* m_library_handle = nullptr;
    ov_genai_ssd_plugin_t* m_plugin = nullptr;
    void* m_instance_handle = nullptr;
    bool m_shut_down = false;
    std::vector<std::uint64_t> m_recovered_hashes;
};

}  // namespace ov::genai
