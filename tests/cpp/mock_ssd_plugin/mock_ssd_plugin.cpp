// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Minimal, dependency-free mock vendor SSD plugin used to test KVCacheOffloadSSDPluginBackend's dynamic
// loading, read/write delegation, and fault isolation (see new_plan.md Phase 4). Deliberately does not
// link against OpenVINO or openvino.genai: a real vendor plugin only needs to compile
// openvino/genai/c/ssd_plugin_interface.h.
//
// Test-only failure injection: storage_plugin_properties may contain "fail_init=1", "fail_write=1", or
// "fail_read=1" (parsed out of the ABI's "key=value;..." properties_string) to exercise the fault paths a
// real vendor plugin could hit.

#include "openvino/genai/c/ssd_plugin_interface.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct MockInstance {
    std::string storage_path;
    // hash -> per-layer bytes.
    std::unordered_map<uint64_t, std::vector<std::vector<uint8_t>>> blocks;
    bool fail_write = false;
    bool fail_read = false;
};

bool properties_has_flag(const char* properties_string, const char* flag) {
    return properties_string != nullptr && std::string(properties_string).find(flag) != std::string::npos;
}

int mock_init(void* handle,
              const char* storage_path,
              const ov_genai_storage_metadata_t* metadata,
              const char* properties_string) {
    (void)metadata;
    auto* instance = static_cast<MockInstance*>(handle);
    if (properties_has_flag(properties_string, "fail_init=1")) {
        return 1;
    }
    instance->storage_path = storage_path != nullptr ? storage_path : "";
    instance->fail_write = properties_has_flag(properties_string, "fail_write=1");
    instance->fail_read = properties_has_flag(properties_string, "fail_read=1");
    return 0;
}

int mock_has_block(void* handle, uint64_t block_hash, int* out_present) {
    auto* instance = static_cast<MockInstance*>(handle);
    *out_present = instance->blocks.count(block_hash) != 0 ? 1 : 0;
    return 0;
}

int mock_has_blocks(void* handle, const uint64_t* block_hashes, size_t count, int* out_present) {
    auto* instance = static_cast<MockInstance*>(handle);
    for (size_t i = 0; i < count; ++i) {
        out_present[i] = instance->blocks.count(block_hashes[i]) != 0 ? 1 : 0;
    }
    return 0;
}

int mock_write_blocks(void* handle, const ov_genai_block_io_desc_t* descriptors, size_t count) {
    auto* instance = static_cast<MockInstance*>(handle);
    if (instance->fail_write) {
        return 2;
    }
    for (size_t i = 0; i < count; ++i) {
        auto& layers = instance->blocks[descriptors[i].block_hash];
        if (layers.size() <= descriptors[i].layer_idx) {
            layers.resize(descriptors[i].layer_idx + 1);
        }
        const auto* data = static_cast<const uint8_t*>(descriptors[i].buffer);
        layers[descriptors[i].layer_idx].assign(data, data + descriptors[i].buffer_bytes);
    }
    return 0;
}

int mock_read_blocks(void* handle, const ov_genai_block_io_desc_t* descriptors, size_t count) {
    auto* instance = static_cast<MockInstance*>(handle);
    if (instance->fail_read) {
        return 3;
    }
    for (size_t i = 0; i < count; ++i) {
        auto it = instance->blocks.find(descriptors[i].block_hash);
        if (it == instance->blocks.end() || it->second.size() <= descriptors[i].layer_idx) {
            return 4;
        }
        const auto& layer_data = it->second[descriptors[i].layer_idx];
        if (layer_data.size() != descriptors[i].buffer_bytes) {
            return 5;
        }
        std::copy(layer_data.begin(), layer_data.end(), static_cast<uint8_t*>(descriptors[i].buffer));
    }
    return 0;
}

int mock_evict_blocks(void* handle, const uint64_t* block_hashes, size_t count) {
    auto* instance = static_cast<MockInstance*>(handle);
    for (size_t i = 0; i < count; ++i) {
        instance->blocks.erase(block_hashes[i]);
    }
    return 0;
}

int mock_flush_manifest(void* handle) {
    (void)handle;
    return 0;
}

void mock_shutdown(void* handle) {
    (void)handle;
}

int mock_get_num_slots(void* handle, size_t* out_num_slots) {
    (void)handle;
    // No real capacity limit: this mock only exists to prove the dynamic-loading/delegation seam works.
    *out_num_slots = 1024;
    return 0;
}

int mock_get_num_free_slots(void* handle, size_t* out_num_free_slots) {
    auto* instance = static_cast<MockInstance*>(handle);
    *out_num_free_slots = 1024 - instance->blocks.size();
    return 0;
}

int mock_get_recovered_hashes(void* handle, uint64_t* out_hashes, size_t max_count, size_t* out_count) {
    // This mock never persists across process restarts, so nothing is ever "recovered".
    (void)handle;
    (void)out_hashes;
    (void)max_count;
    *out_count = 0;
    return 0;
}

void mock_describe(void* handle, char* buffer, size_t buffer_size) {
    auto* instance = static_cast<MockInstance*>(handle);
    if (buffer_size == 0) {
        return;
    }
    const std::string description = "MockSSDPlugin(" + instance->storage_path + ")";
    const size_t to_copy = std::min(description.size(), buffer_size - 1);
    std::memcpy(buffer, description.data(), to_copy);
    buffer[to_copy] = '\0';
}

void* mock_create_instance() {
    return new MockInstance();
}

void mock_destroy_instance(void* handle) {
    delete static_cast<MockInstance*>(handle);
}

ov_genai_ssd_plugin_t g_plugin = {
#ifdef MOCK_SSD_PLUGIN_BAD_API_VERSION
    OV_GENAI_SSD_PLUGIN_API_VERSION + 1,  // deliberately incompatible, for version-mismatch tests
#else
    OV_GENAI_SSD_PLUGIN_API_VERSION,
#endif
    "MockVendor",
    mock_create_instance,
    mock_destroy_instance,
    mock_init,
    mock_has_block,
    mock_has_blocks,
    mock_write_blocks,
    mock_read_blocks,
    mock_evict_blocks,
    mock_flush_manifest,
    mock_shutdown,
    mock_get_num_slots,
    mock_get_num_free_slots,
    mock_get_recovered_hashes,
    mock_describe,
};

}  // namespace

extern "C" {

#ifdef _WIN32
__declspec(dllexport)
#endif
ov_genai_ssd_plugin_t*
ov_genai_get_ssd_plugin(void) {
    return &g_plugin;
}

}  // extern "C"
