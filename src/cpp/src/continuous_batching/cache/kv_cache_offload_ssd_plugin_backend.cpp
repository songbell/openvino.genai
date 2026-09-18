// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/kv_cache_offload_ssd_plugin_backend.hpp"

#include <sstream>

#include "openvino/core/except.hpp"

#ifdef _WIN32
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace ov::genai {

namespace {

/// Builds the C ABI's lossy, uniform-layer-size metadata struct from the caller's full layout, throwing
/// if the model's layers are not actually uniform (see the class doc comment for why plugins can't
/// express that case).
StorageModelMetadata build_metadata(const KVCacheDiskLayout& layout, const CacheOffloadConfig& config) {
    StorageModelMetadata metadata;
    metadata.model_fingerprint = config.model_fingerprint;
    metadata.tokenizer_hash = config.tokenizer_fingerprint;
    metadata.num_layers = layout.get_num_layers();
    OPENVINO_ASSERT(metadata.num_layers > 0, "KV cache offload plugin backend requires at least one decoder layer");

    const size_t first_layer_bytes = layout.get_key_segment(0).size + layout.get_value_segment(0).size;
    for (size_t layer = 1; layer < metadata.num_layers; ++layer) {
        const size_t layer_bytes = layout.get_key_segment(layer).size + layout.get_value_segment(layer).size;
        OPENVINO_ASSERT(layer_bytes == first_layer_bytes,
                        "KV cache offload plugin backends require every decoder layer to have the same "
                        "combined key+value byte size; layer 0 has ",
                        first_layer_bytes,
                        " bytes, layer ",
                        layer,
                        " has ",
                        layer_bytes,
                        " bytes");
    }
    metadata.block_bytes_per_layer = first_layer_bytes;
    return metadata;
}

/// Serializes an `ov::AnyMap` into `key1=value1;key2=value2;...` for the C ABI's opaque
/// `properties_string` (see the deviation note in ssd_plugin_interface.h for why not JSON).
std::string stringify_properties(const ov::AnyMap& properties) {
    std::ostringstream oss;
    bool first = true;
    for (const auto& [key, value] : properties) {
        if (!first) {
            oss << ";";
        }
        first = false;
        try {
            oss << key << "=" << value.as<std::string>();
        } catch (const std::exception& error) {
            OPENVINO_THROW("KV cache offload plugin property '",
                           key,
                           "' could not be converted to a string: ",
                           error.what());
        }
    }
    return oss.str();
}

}  // namespace

KVCacheOffloadSSDPluginBackend::KVCacheOffloadSSDPluginBackend(const KVCacheDiskLayout& layout,
                                                               const CacheOffloadConfig& config)
    : m_plugin_path(config.storage_plugin_path) {
    OPENVINO_ASSERT(!m_plugin_path.empty(),
                    "KV cache offload storage_backend_type is 'plugin' but storage_plugin_path is empty");
    try {
        load_library(m_plugin_path);

        m_instance_handle = m_plugin->create_instance();
        OPENVINO_ASSERT(m_instance_handle != nullptr,
                        "KV cache offload plugin '",
                        m_plugin_path,
                        "' create_instance() returned null");

        const auto metadata = build_metadata(layout, config);
        initialize(config.storage_cache_dir, metadata, config.storage_plugin_properties);
    } catch (...) {
        if (m_plugin != nullptr && m_instance_handle != nullptr) {
            m_plugin->destroy_instance(m_instance_handle);
            m_instance_handle = nullptr;
        }
        unload_library();
        throw;
    }
}

KVCacheOffloadSSDPluginBackend::~KVCacheOffloadSSDPluginBackend() {
    shutdown();
    unload_library();
}

void KVCacheOffloadSSDPluginBackend::load_library(const std::string& plugin_path) {
    ov_genai_get_ssd_plugin_fn get_plugin_fn = nullptr;
#ifdef _WIN32
    HMODULE handle = LoadLibraryA(plugin_path.c_str());
    OPENVINO_ASSERT(handle != nullptr,
                    "Failed to load KV cache offload storage plugin '",
                    plugin_path,
                    "' (LoadLibrary error ",
                    GetLastError(),
                    ")");
    m_library_handle = handle;
    get_plugin_fn = reinterpret_cast<ov_genai_get_ssd_plugin_fn>(GetProcAddress(handle, "ov_genai_get_ssd_plugin"));
#else
    void* handle = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    OPENVINO_ASSERT(handle != nullptr,
                    "Failed to load KV cache offload storage plugin '",
                    plugin_path,
                    "': ",
                    dlerror());
    m_library_handle = handle;
    get_plugin_fn = reinterpret_cast<ov_genai_get_ssd_plugin_fn>(dlsym(handle, "ov_genai_get_ssd_plugin"));
#endif

    if (get_plugin_fn == nullptr) {
        unload_library();
        OPENVINO_THROW("KV cache offload storage plugin '",
                       plugin_path,
                       "' does not export the required 'ov_genai_get_ssd_plugin' symbol");
    }

    m_plugin = get_plugin_fn();
    if (m_plugin == nullptr) {
        unload_library();
        OPENVINO_THROW("KV cache offload storage plugin '", plugin_path, "' ov_genai_get_ssd_plugin() returned null");
    }
    if (m_plugin->api_version != OV_GENAI_SSD_PLUGIN_API_VERSION) {
        const uint32_t got_version = m_plugin->api_version;
        m_plugin = nullptr;
        unload_library();
        OPENVINO_THROW("KV cache offload storage plugin '",
                       plugin_path,
                       "' has an incompatible API version (expected ",
                       OV_GENAI_SSD_PLUGIN_API_VERSION,
                       ", got ",
                       got_version,
                       ")");
    }
}

void KVCacheOffloadSSDPluginBackend::unload_library() noexcept {
    m_plugin = nullptr;
    if (m_library_handle == nullptr) {
        return;
    }
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(m_library_handle));
#else
    dlclose(m_library_handle);
#endif
    m_library_handle = nullptr;
}

void KVCacheOffloadSSDPluginBackend::initialize(const std::string& storage_path,
                                                const StorageModelMetadata& metadata,
                                                const ov::AnyMap& custom_properties) {
    ov_genai_storage_metadata_t c_metadata{};
    c_metadata.model_name = metadata.model_name.c_str();
    c_metadata.model_fingerprint = metadata.model_fingerprint.c_str();
    c_metadata.tokenizer_hash = metadata.tokenizer_hash.c_str();
    c_metadata.kv_block_size_tokens = metadata.kv_block_size_tokens;
    c_metadata.num_layers = metadata.num_layers;
    c_metadata.block_bytes_per_layer = metadata.block_bytes_per_layer;

    const std::string properties_string = stringify_properties(custom_properties);
    const int init_status = m_plugin->init(m_instance_handle, storage_path.c_str(), &c_metadata, properties_string.c_str());
    OPENVINO_ASSERT(init_status == 0,
                    "KV cache offload plugin '",
                    m_plugin_path,
                    "' init() failed with status ",
                    init_status);

    size_t recovered_count = 0;
    int status = m_plugin->get_recovered_hashes(m_instance_handle, nullptr, 0, &recovered_count);
    OPENVINO_ASSERT(status == 0,
                    "KV cache offload plugin '",
                    m_plugin_path,
                    "' get_recovered_hashes() count query failed with status ",
                    status);
    m_recovered_hashes.assign(recovered_count, 0);
    if (recovered_count > 0) {
        size_t filled_count = 0;
        status = m_plugin->get_recovered_hashes(m_instance_handle, m_recovered_hashes.data(), recovered_count, &filled_count);
        OPENVINO_ASSERT(status == 0 && filled_count == recovered_count,
                        "KV cache offload plugin '",
                        m_plugin_path,
                        "' get_recovered_hashes() fill failed");
    }
}

bool KVCacheOffloadSSDPluginBackend::has_block(std::uint64_t block_hash) const {
    int present = 0;
    const int status = m_plugin->has_block(m_instance_handle, block_hash, &present);
    OPENVINO_ASSERT(status == 0, "KV cache offload plugin '", m_plugin_path, "' has_block() failed with status ", status);
    return present != 0;
}

std::vector<bool> KVCacheOffloadSSDPluginBackend::has_blocks(const std::vector<std::uint64_t>& block_hashes) const {
    std::vector<bool> result(block_hashes.size());
    if (block_hashes.empty()) {
        return result;
    }
    std::vector<int> present(block_hashes.size(), 0);
    const int status = m_plugin->has_blocks(m_instance_handle, block_hashes.data(), block_hashes.size(), present.data());
    OPENVINO_ASSERT(status == 0, "KV cache offload plugin '", m_plugin_path, "' has_blocks() failed with status ", status);
    for (size_t i = 0; i < block_hashes.size(); ++i) {
        result[i] = present[i] != 0;
    }
    return result;
}

bool KVCacheOffloadSSDPluginBackend::write_blocks(const std::vector<BlockIORequest>& requests) {
    if (requests.empty()) {
        return true;
    }
    std::vector<ov_genai_block_io_desc_t> descriptors(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        descriptors[i].block_hash = requests[i].block_hash;
        descriptors[i].buffer = requests[i].host_buffer;
        descriptors[i].buffer_bytes = requests[i].buffer_size_bytes;
        descriptors[i].layer_idx = requests[i].layer_idx;
    }
    return m_plugin->write_blocks(m_instance_handle, descriptors.data(), descriptors.size()) == 0;
}

bool KVCacheOffloadSSDPluginBackend::read_blocks(const std::vector<BlockIORequest>& requests) {
    if (requests.empty()) {
        return true;
    }
    std::vector<ov_genai_block_io_desc_t> descriptors(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        descriptors[i].block_hash = requests[i].block_hash;
        descriptors[i].buffer = requests[i].host_buffer;
        descriptors[i].buffer_bytes = requests[i].buffer_size_bytes;
        descriptors[i].layer_idx = requests[i].layer_idx;
    }
    return m_plugin->read_blocks(m_instance_handle, descriptors.data(), descriptors.size()) == 0;
}

void KVCacheOffloadSSDPluginBackend::evict_blocks(const std::vector<std::uint64_t>& block_hashes) {
    if (block_hashes.empty()) {
        return;
    }
    const int status = m_plugin->evict_blocks(m_instance_handle, block_hashes.data(), block_hashes.size());
    OPENVINO_ASSERT(status == 0, "KV cache offload plugin '", m_plugin_path, "' evict_blocks() failed with status ", status);
}

void KVCacheOffloadSSDPluginBackend::flush_manifest() {
    const int status = m_plugin->flush_manifest(m_instance_handle);
    OPENVINO_ASSERT(status == 0, "KV cache offload plugin '", m_plugin_path, "' flush_manifest() failed with status ", status);
}

void KVCacheOffloadSSDPluginBackend::shutdown() {
    if (m_shut_down) {
        return;
    }
    m_shut_down = true;
    if (m_plugin != nullptr && m_instance_handle != nullptr) {
        m_plugin->shutdown(m_instance_handle);
        m_plugin->destroy_instance(m_instance_handle);
        m_instance_handle = nullptr;
    }
}

std::size_t KVCacheOffloadSSDPluginBackend::get_num_slots() const {
    size_t num_slots = 0;
    const int status = m_plugin->get_num_slots(m_instance_handle, &num_slots);
    OPENVINO_ASSERT(status == 0, "KV cache offload plugin '", m_plugin_path, "' get_num_slots() failed with status ", status);
    return num_slots;
}

std::size_t KVCacheOffloadSSDPluginBackend::get_num_free_slots() const {
    size_t num_free_slots = 0;
    const int status = m_plugin->get_num_free_slots(m_instance_handle, &num_free_slots);
    OPENVINO_ASSERT(status == 0,
                    "KV cache offload plugin '",
                    m_plugin_path,
                    "' get_num_free_slots() failed with status ",
                    status);
    return num_free_slots;
}

std::string KVCacheOffloadSSDPluginBackend::describe() const {
    char buffer[256] = {};
    m_plugin->describe(m_instance_handle, buffer, sizeof(buffer));
    return std::string(buffer);
}

}  // namespace ov::genai
