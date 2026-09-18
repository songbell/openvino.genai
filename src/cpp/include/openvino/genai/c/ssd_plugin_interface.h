// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Pure C ABI a third-party SSD storage plugin shared library must export so it can be loaded by
// openvino.genai without linking against its C++ ABI (see new_plan.md Phase 4 / SS3.3.2). A plugin
// only needs to compile this header and export the `ov_genai_get_ssd_plugin` symbol below; it does not
// need to link the OpenVINO or openvino.genai C++ libraries at all.
//
// Deviations from the illustrative interface in new_plan.md SS3.3.2, chosen for a minimal, dependency-free
// v1 ABI:
//   - `properties_string` replaces a JSON blob: openvino.genai serializes CacheOffloadConfig's
//     `storage_plugin_properties` (an `ov::AnyMap`) as `key1=value1;key2=value2;...` (values must be
//     convertible to string) rather than pulling in a JSON library on either side of the ABI boundary.
//     A vendor plugin that needs richer structured configuration can parse its own format out of this
//     string, or ignore it.
//   - `get_recovered_hashes` uses a two-call size-then-fill pattern (call with `max_count == 0` first to
//     get the count, then again with a large-enough buffer) instead of returning an owned array, since a
//     C ABI cannot safely hand back a heap allocation for the caller to free with a possibly different
//     allocator.

#ifndef OPENVINO_GENAI_C_SSD_PLUGIN_INTERFACE_H
#define OPENVINO_GENAI_C_SSD_PLUGIN_INTERFACE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bumped whenever a breaking change is made to `ov_genai_ssd_plugin_t`'s layout; a loader must reject
// any plugin whose `api_version` does not match exactly.
#define OV_GENAI_SSD_PLUGIN_API_VERSION 0x00010000u

typedef struct {
    const char* model_name;
    const char* model_fingerprint;
    const char* tokenizer_hash;
    size_t kv_block_size_tokens;
    size_t num_layers;
    // Combined key+value bytes for one layer of one block. Assumes every layer has the same byte size;
    // a caller-side model with non-uniform per-layer sizes cannot use a pluggable backend (see the
    // in-tree DefaultFileStorageBackend, which does not have this restriction because it is handed the
    // full per-layer layout directly rather than through this lossy, uniform-layer-size struct).
    size_t block_bytes_per_layer;
} ov_genai_storage_metadata_t;

typedef struct {
    uint64_t block_hash;
    // Owned by the caller; a plugin must not free it, and must not retain the pointer past the call.
    void* buffer;
    size_t buffer_bytes;
    uint32_t layer_idx;
} ov_genai_block_io_desc_t;

// Every function pointer below returns 0 on success and a non-zero, plugin-defined status code on
// failure (openvino.genai treats any non-zero return as an operation failure and does not try to
// interpret the value); pointers besides `create_instance`/`destroy_instance` receive the `void*` handle
// `create_instance` returned.
typedef struct ov_genai_ssd_plugin {
    uint32_t api_version;  // Must equal OV_GENAI_SSD_PLUGIN_API_VERSION.
    const char* vendor_name;

    void* (*create_instance)(void);
    void (*destroy_instance)(void* handle);

    int (*init)(void* handle,
               const char* storage_path,
               const ov_genai_storage_metadata_t* metadata,
               const char* properties_string);

    int (*has_block)(void* handle, uint64_t block_hash, int* out_present);
    // `out_present` must have room for `count` entries.
    int (*has_blocks)(void* handle, const uint64_t* block_hashes, size_t count, int* out_present);

    int (*write_blocks)(void* handle, const ov_genai_block_io_desc_t* descriptors, size_t count);
    int (*read_blocks)(void* handle, const ov_genai_block_io_desc_t* descriptors, size_t count);
    int (*evict_blocks)(void* handle, const uint64_t* block_hashes, size_t count);
    int (*flush_manifest)(void* handle);
    void (*shutdown)(void* handle);

    int (*get_num_slots)(void* handle, size_t* out_num_slots);
    int (*get_num_free_slots)(void* handle, size_t* out_num_free_slots);
    // Two-call pattern: call with max_count == 0 (out_hashes may be NULL) to get the count into
    // *out_count without writing to out_hashes, then call again with a large-enough buffer to fill it.
    int (*get_recovered_hashes)(void* handle, uint64_t* out_hashes, size_t max_count, size_t* out_count);

    // Writes a NUL-terminated description into `buffer` (truncated to fit `buffer_size` including the
    // terminator), for logging/diagnostics only.
    void (*describe)(void* handle, char* buffer, size_t buffer_size);
} ov_genai_ssd_plugin_t;

// A plugin shared library must export exactly one symbol with this name and signature; openvino.genai
// resolves it by name (`dlsym`/`GetProcAddress`) after loading the library.
typedef ov_genai_ssd_plugin_t* (*ov_genai_get_ssd_plugin_fn)(void);

#ifdef __cplusplus
}
#endif

#endif  // OPENVINO_GENAI_C_SSD_PLUGIN_INTERFACE_H
