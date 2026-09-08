// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "continuous_batching/cache/kv_cache_offload_ssd_plugin_backend.hpp"

#ifdef _WIN32
#    include <windows.h>
#else
#    include <limits.h>
#    include <unistd.h>
#endif

using namespace ov::genai;

namespace {

constexpr size_t KEY_BLOCK_BYTES = 32;
constexpr size_t VALUE_BLOCK_BYTES = 32;
constexpr size_t NUM_LAYERS = 2;
constexpr size_t SLOT_BYTES = NUM_LAYERS * (KEY_BLOCK_BYTES + VALUE_BLOCK_BYTES);

KVCacheDiskLayout make_layout() {
    return KVCacheDiskLayout(std::vector<size_t>(NUM_LAYERS, KEY_BLOCK_BYTES),
                             std::vector<size_t>(NUM_LAYERS, VALUE_BLOCK_BYTES));
}

/// @return The directory this test executable itself lives in, where CMake also places the mock plugin
/// shared libraries built alongside it (see tests/cpp/mock_ssd_plugin/CMakeLists.txt).
std::filesystem::path executable_directory() {
#ifdef _WIN32
    char buffer[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    return std::filesystem::path(buffer).parent_path();
#else
    char buffer[PATH_MAX] = {};
    const ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length <= 0) {
        return std::filesystem::current_path();
    }
    return std::filesystem::path(std::string(buffer, static_cast<size_t>(length))).parent_path();
#endif
}

std::filesystem::path shared_library_path(const std::string& base_name) {
#ifdef _WIN32
    return executable_directory() / (base_name + ".dll");
#else
    return executable_directory() / ("lib" + base_name + ".so");
#endif
}

std::filesystem::path mock_plugin_path() {
    return shared_library_path("mock_ssd_plugin");
}

std::filesystem::path mock_plugin_bad_version_path() {
    return shared_library_path("mock_ssd_plugin_bad_version");
}

CacheOffloadConfig make_plugin_config(const std::filesystem::path& plugin_path) {
    CacheOffloadConfig config;
    config.storage_backend_type = "plugin";
    config.storage_plugin_path = plugin_path.string();
    config.model_fingerprint = "model-a";
    config.tokenizer_fingerprint = "tokenizer-a";
    return config;
}

std::vector<uint8_t> make_pattern(uint8_t seed) {
    std::vector<uint8_t> data(SLOT_BYTES);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(seed + i);
    }
    return data;
}

std::vector<BlockIORequest> make_requests(uint64_t hash, std::vector<uint8_t>& buffer) {
    const auto layout = make_layout();
    std::vector<BlockIORequest> requests;
    for (size_t layer = 0; layer < layout.get_num_layers(); ++layer) {
        const auto key = layout.get_key_segment(layer);
        const auto value = layout.get_value_segment(layer);
        BlockIORequest request;
        request.block_hash = hash;
        request.host_buffer = buffer.data() + key.offset;
        request.buffer_size_bytes = key.size + value.size;
        request.layer_idx = static_cast<uint32_t>(layer);
        requests.push_back(request);
    }
    return requests;
}

}  // namespace

TEST(TestKVCacheOffloadSSDPluginBackend, MockPluginLibrariesWereBuilt) {
    // If these are missing, tests/cpp/mock_ssd_plugin was not built alongside this executable; every
    // other test in this file will fail with a confusing "failed to load" error instead of this one clear
    // signal.
    ASSERT_TRUE(std::filesystem::exists(mock_plugin_path())) << mock_plugin_path();
    ASSERT_TRUE(std::filesystem::exists(mock_plugin_bad_version_path())) << mock_plugin_bad_version_path();
    ASSERT_TRUE(std::filesystem::exists(shared_library_path("not_a_plugin"))) << shared_library_path("not_a_plugin");
}

TEST(TestKVCacheOffloadSSDPluginBackend, ThrowsWhenPluginFileDoesNotExist) {
    CacheOffloadConfig config = make_plugin_config(executable_directory() / "ov_genai_nonexistent_plugin.dll");
    EXPECT_THROW(KVCacheOffloadSSDPluginBackend(make_layout(), config), ov::Exception);
}

TEST(TestKVCacheOffloadSSDPluginBackend, ThrowsWhenLibraryDoesNotExportEntrySymbol) {
    // A real shared library that loads fine but is not an SSD plugin (does not export
    // ov_genai_get_ssd_plugin): proves the loader distinguishes "can't load the file" from
    // "loaded fine but isn't a valid plugin".
    const auto not_a_plugin = shared_library_path("not_a_plugin");
    ASSERT_TRUE(std::filesystem::exists(not_a_plugin)) << not_a_plugin;
    CacheOffloadConfig config = make_plugin_config(not_a_plugin);
    EXPECT_THROW(KVCacheOffloadSSDPluginBackend(make_layout(), config), ov::Exception);
}

TEST(TestKVCacheOffloadSSDPluginBackend, ThrowsOnApiVersionMismatch) {
    CacheOffloadConfig config = make_plugin_config(mock_plugin_bad_version_path());
    EXPECT_THROW(KVCacheOffloadSSDPluginBackend(make_layout(), config), ov::Exception);
}

TEST(TestKVCacheOffloadSSDPluginBackend, ThrowsWhenPluginInitFails) {
    CacheOffloadConfig config = make_plugin_config(mock_plugin_path());
    config.storage_plugin_properties = {{"fail_init", "1"}};
    EXPECT_THROW(KVCacheOffloadSSDPluginBackend(make_layout(), config), ov::Exception);
}

TEST(TestKVCacheOffloadSSDPluginBackend, LoadsAndRoundTripsThroughTheRealDynamicLibrary) {
    CacheOffloadConfig config = make_plugin_config(mock_plugin_path());
    KVCacheOffloadSSDPluginBackend backend(make_layout(), config);

    EXPECT_FALSE(backend.has_block(42));
    auto data = make_pattern(5);
    ASSERT_TRUE(backend.write_blocks(make_requests(42, data)));
    EXPECT_TRUE(backend.has_block(42));

    std::vector<uint8_t> actual(SLOT_BYTES);
    ASSERT_TRUE(backend.read_blocks(make_requests(42, actual)));
    EXPECT_EQ(actual, data);
}

TEST(TestKVCacheOffloadSSDPluginBackend, HasBlocksBatchQueryMatchesPerHashState) {
    CacheOffloadConfig config = make_plugin_config(mock_plugin_path());
    KVCacheOffloadSSDPluginBackend backend(make_layout(), config);

    auto data = make_pattern(1);
    ASSERT_TRUE(backend.write_blocks(make_requests(1, data)));

    const auto result = backend.has_blocks({1, 2});
    ASSERT_EQ(result.size(), 2u);
    EXPECT_TRUE(result[0]);
    EXPECT_FALSE(result[1]);
}

TEST(TestKVCacheOffloadSSDPluginBackend, EvictBlocksRemovesEntryFromThePlugin) {
    CacheOffloadConfig config = make_plugin_config(mock_plugin_path());
    KVCacheOffloadSSDPluginBackend backend(make_layout(), config);

    auto data = make_pattern(2);
    ASSERT_TRUE(backend.write_blocks(make_requests(7, data)));
    ASSERT_TRUE(backend.has_block(7));

    backend.evict_blocks({7});
    EXPECT_FALSE(backend.has_block(7));
}

TEST(TestKVCacheOffloadSSDPluginBackend, WriteFailurePropagatesAsFalseNotAThrow) {
    CacheOffloadConfig config = make_plugin_config(mock_plugin_path());
    config.storage_plugin_properties = {{"fail_write", "1"}};
    KVCacheOffloadSSDPluginBackend backend(make_layout(), config);

    auto data = make_pattern(3);
    EXPECT_FALSE(backend.write_blocks(make_requests(9, data)));
}

TEST(TestKVCacheOffloadSSDPluginBackend, ReadFailurePropagatesAsFalseNotAThrow) {
    CacheOffloadConfig config = make_plugin_config(mock_plugin_path());
    config.storage_plugin_properties = {{"fail_read", "1"}};
    KVCacheOffloadSSDPluginBackend backend(make_layout(), config);

    std::vector<uint8_t> actual(SLOT_BYTES);
    EXPECT_FALSE(backend.read_blocks(make_requests(1, actual)));
}

TEST(TestKVCacheOffloadSSDPluginBackend, DescribeReturnsThePluginSuppliedString) {
    CacheOffloadConfig config = make_plugin_config(mock_plugin_path());
    KVCacheOffloadSSDPluginBackend backend(make_layout(), config);
    EXPECT_NE(backend.describe().find("MockSSDPlugin"), std::string::npos);
}

TEST(TestKVCacheOffloadSSDPluginBackend, NonUniformLayerSizesAreRejectedUpFront) {
    // ov_genai_storage_metadata_t::block_bytes_per_layer is a single scalar (see
    // ssd_plugin_interface.h): a plugin backend cannot serve a model whose decoder layers have
    // different combined key+value byte sizes.
    KVCacheDiskLayout non_uniform_layout(std::vector<size_t>{KEY_BLOCK_BYTES, KEY_BLOCK_BYTES + 8},
                                        std::vector<size_t>{VALUE_BLOCK_BYTES, VALUE_BLOCK_BYTES});
    CacheOffloadConfig config = make_plugin_config(mock_plugin_path());
    EXPECT_THROW(KVCacheOffloadSSDPluginBackend(non_uniform_layout, config), ov::Exception);
}
