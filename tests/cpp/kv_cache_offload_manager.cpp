// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>

#include "continuous_batching/cache/kv_cache_isolation_seed.hpp"

#include "continuous_batching/cache/kv_cache_offload_manager.hpp"

using namespace ov::genai;

namespace {

constexpr size_t KEY_BLOCK_BYTES = 96;
constexpr size_t VALUE_BLOCK_BYTES = 64;
constexpr size_t NUM_LAYERS = 3;
constexpr size_t SLOT_BYTES = NUM_LAYERS * (KEY_BLOCK_BYTES + VALUE_BLOCK_BYTES);

KVCacheDiskLayout make_layout() {
    return KVCacheDiskLayout(std::vector<size_t>(NUM_LAYERS, KEY_BLOCK_BYTES),
                             std::vector<size_t>(NUM_LAYERS, VALUE_BLOCK_BYTES));
}

CacheOffloadConfig make_config(size_t num_slots) {
    CacheOffloadConfig config;
    config.capacity_bytes = num_slots * SLOT_BYTES;
    return config;
}

std::vector<uint8_t> make_pattern(uint8_t seed) {
    std::vector<uint8_t> data(SLOT_BYTES);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(seed + i);
    }
    return data;
}

/// Builds one BlockIORequest per layer for @p hash, pointing into (sub-spans of) @p buffer, which must
/// already be sized to SLOT_BYTES. Mirrors the request-construction pattern KVCacheOffloadCache itself
/// uses, since the interface requires every layer of a block to be submitted together in one call.
std::vector<BlockIORequest> make_requests(uint64_t hash,
                                          std::vector<uint8_t>& buffer,
                                          const KVCacheDiskLayout& layout = make_layout()) {
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

TEST(TestKVCacheOffloadManager, DerivesSlotCountFromCapacity) {
    // Capacity that is not a multiple of the slot size must be rounded down.
    KVCacheOffloadManager manager(make_layout(), make_config(4));

    EXPECT_EQ(manager.get_num_slots(), 4);
    EXPECT_EQ(manager.get_num_free_slots(), 4);
}

TEST(TestKVCacheOffloadManager, DescribesBufferBudgetAndDefaults) {
    CacheOffloadConfig config;

    EXPECT_EQ(config.buffer_slots, 2);
    EXPECT_TRUE(config.to_string().find("buffer_slots: 2") != std::string::npos);
    EXPECT_TRUE(config.to_string().find("wait_for_buffer: false") != std::string::npos);
    EXPECT_TRUE(config.to_string().find("enable_detailed_logging: false") != std::string::npos);
}

TEST(TestKVCacheOffloadManager, CreatesAndRemovesRunSpecificFile) {
    std::filesystem::path file_path;
    {
        KVCacheOffloadManager manager(make_layout(), make_config(2));
        file_path = manager.describe();

        ASSERT_FALSE(file_path.empty());
        EXPECT_TRUE(std::filesystem::exists(file_path));
        EXPECT_EQ(std::filesystem::file_size(file_path), 2 * SLOT_BYTES);
    }
    EXPECT_FALSE(std::filesystem::exists(file_path));
}

TEST(TestKVCacheOffloadManager, HashRoundTrip) {
    KVCacheOffloadManager manager(make_layout(), make_config(2));
    auto expected = make_pattern(7);

    ASSERT_TRUE(manager.write_blocks(make_requests(1, expected)));

    std::vector<uint8_t> actual(SLOT_BYTES);
    ASSERT_TRUE(manager.read_blocks(make_requests(1, actual)));
    EXPECT_EQ(actual, expected);
}

TEST(TestKVCacheOffloadManager, BlocksDoNotOverlap) {
    KVCacheOffloadManager manager(make_layout(), make_config(3));
    auto first = make_pattern(1);
    auto second = make_pattern(200);

    ASSERT_TRUE(manager.write_blocks(make_requests(10, first)));
    ASSERT_TRUE(manager.write_blocks(make_requests(20, second)));

    std::vector<uint8_t> actual(SLOT_BYTES);
    ASSERT_TRUE(manager.read_blocks(make_requests(10, actual)));
    EXPECT_EQ(actual, first);
    ASSERT_TRUE(manager.read_blocks(make_requests(20, actual)));
    EXPECT_EQ(actual, second);
}

TEST(TestKVCacheOffloadManager, WriteExhaustsFreeSlotsAndEvictFreesThem) {
    KVCacheOffloadManager manager(make_layout(), make_config(2));

    auto first = make_pattern(1);
    auto second = make_pattern(2);
    ASSERT_TRUE(manager.write_blocks(make_requests(1, first)));
    ASSERT_TRUE(manager.write_blocks(make_requests(2, second)));
    EXPECT_EQ(manager.get_num_free_slots(), 0u);

    auto third = make_pattern(3);
    EXPECT_THROW(manager.write_blocks(make_requests(3, third)), ov::Exception);

    manager.evict_blocks({1});
    EXPECT_EQ(manager.get_num_free_slots(), 1u);
    ASSERT_TRUE(manager.write_blocks(make_requests(3, third)));
    EXPECT_EQ(manager.get_num_free_slots(), 0u);
}

TEST(TestKVCacheOffloadManager, EvictingUnknownHashIsNoOp) {
    KVCacheOffloadManager manager(make_layout(), make_config(2));
    EXPECT_NO_THROW(manager.evict_blocks({123}));
}

TEST(TestKVCacheOffloadManager, RejectsWrongLayerRequestSize) {
    KVCacheOffloadManager manager(make_layout(), make_config(2));
    auto data = make_pattern(0);
    auto requests = make_requests(1, data);
    requests[0].buffer_size_bytes -= 1;  // wrong size for that layer.
    EXPECT_THROW(manager.write_blocks(requests), ov::Exception);
}

TEST(TestKVCacheOffloadManager, MissingHashOnReadIsAMissNotAThrow) {
    KVCacheOffloadManager manager(make_layout(), make_config(2));
    std::vector<uint8_t> block_data(SLOT_BYTES);
    EXPECT_FALSE(manager.read_blocks(make_requests(123, block_data)));
}

TEST(TestKVCacheOffloadManager, RejectsCapacitySmallerThanOneBlock) {
    CacheOffloadConfig config;
    config.capacity_bytes = SLOT_BYTES - 1;

    EXPECT_THROW(KVCacheOffloadManager(make_layout(), config), ov::Exception);
}

TEST(TestKVCacheOffloadManager, IsSupportedDeviceRecognizesCpuAndGpuOnly) {
    EXPECT_TRUE(KVCacheOffloadManager::is_supported_device("CPU"));
    EXPECT_TRUE(KVCacheOffloadManager::is_supported_device("GPU"));
    EXPECT_TRUE(KVCacheOffloadManager::is_supported_device("GPU.0"));
    EXPECT_FALSE(KVCacheOffloadManager::is_supported_device("NPU"));
}

TEST(TestKVCacheOffloadManager, RejectsDirectIO) {
    CacheOffloadConfig direct_io = make_config(1);
    direct_io.use_page_cache = false;
    EXPECT_THROW(KVCacheOffloadManager(make_layout(), direct_io), ov::Exception);
}

TEST(TestKVCacheOffloadManager, RejectsMissingDirectory) {
    CacheOffloadConfig config = make_config(1);
    config.storage_cache_dir = (std::filesystem::temp_directory_path() / "ov_genai_offload_missing_dir").string();
    ASSERT_FALSE(std::filesystem::exists(config.storage_cache_dir));

    EXPECT_THROW(KVCacheOffloadManager(make_layout(), config), ov::Exception);
}

TEST(TestKVCacheOffloadManager, UsesConfiguredDirectory) {
    const auto directory = std::filesystem::temp_directory_path() / "ov_genai_offload_test_dir";
    std::filesystem::create_directories(directory);

    CacheOffloadConfig config = make_config(1);
    config.storage_cache_dir = directory.string();

    {
        KVCacheOffloadManager manager(make_layout(), config);
        EXPECT_EQ(std::filesystem::path(manager.describe()).parent_path(), directory);
    }

    std::filesystem::remove_all(directory);
}

namespace {

CacheOffloadConfig make_persistent_config(size_t num_slots, const std::filesystem::path& directory) {
    CacheOffloadConfig config = make_config(num_slots);
    config.storage_cache_dir = directory.string();
    config.enable_persistence = true;
    config.model_fingerprint = "model-a";
    config.tokenizer_fingerprint = "tokenizer-a";
    return config;
}

class PersistentCacheDirFixture : public ::testing::Test {
protected:
    void SetUp() override {
        directory = std::filesystem::temp_directory_path() /
                    ("ov_genai_offload_persist_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
                     "_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
        std::filesystem::create_directories(directory);
    }

    void TearDown() override {
        std::filesystem::remove_all(directory);
    }

    std::filesystem::path directory;
};

}  // namespace

TEST_F(PersistentCacheDirFixture, RequiresPathAndFingerprints) {
    CacheOffloadConfig config = make_config(2);
    config.enable_persistence = true;
    // No path, no fingerprints set.
    EXPECT_THROW(KVCacheOffloadManager(make_layout(), config), ov::Exception);

    config.storage_cache_dir = directory.string();
    // Fingerprints still empty.
    EXPECT_THROW(KVCacheOffloadManager(make_layout(), config), ov::Exception);

    config.model_fingerprint = "model-a";
    // tokenizer_fingerprint still empty.
    EXPECT_THROW(KVCacheOffloadManager(make_layout(), config), ov::Exception);
}

TEST_F(PersistentCacheDirFixture, FreshPersistentCacheHasNoRecoveredEntries) {
    KVCacheOffloadManager manager(make_layout(), make_persistent_config(3, directory));
    EXPECT_TRUE(manager.get_recovered_hashes().empty());
    EXPECT_TRUE(std::filesystem::exists(manager.describe()));
}

TEST_F(PersistentCacheDirFixture, SurvivesDestructionAndIsRecoveredOnReopen) {
    const auto config = make_persistent_config(3, directory);
    auto first_data = make_pattern(11);
    auto second_data = make_pattern(99);

    {
        KVCacheOffloadManager manager(make_layout(), config);
        ASSERT_TRUE(manager.write_blocks(make_requests(42, first_data)));
        ASSERT_TRUE(manager.write_blocks(make_requests(77, second_data)));
        // manager destructs here; files must survive since persistence is enabled.
    }

    ASSERT_TRUE(std::filesystem::exists(directory / "ov_genai_kv_offload.data"));
    ASSERT_TRUE(std::filesystem::exists(directory / "ov_genai_kv_offload.manifest"));

    KVCacheOffloadManager manager(make_layout(), config);
    const auto& recovered = manager.get_recovered_hashes();
    ASSERT_EQ(recovered.size(), 2u);
    EXPECT_NE(std::find(recovered.begin(), recovered.end(), 42u), recovered.end());
    EXPECT_NE(std::find(recovered.begin(), recovered.end(), 77u), recovered.end());

    std::vector<uint8_t> actual(SLOT_BYTES);
    ASSERT_TRUE(manager.read_blocks(make_requests(42, actual)));
    EXPECT_EQ(actual, first_data);
    ASSERT_TRUE(manager.read_blocks(make_requests(77, actual)));
    EXPECT_EQ(actual, second_data);

    // Recovered slots must not be handed out as free slots.
    EXPECT_EQ(manager.get_num_free_slots(), 1u);
}

TEST_F(PersistentCacheDirFixture, MismatchedModelFingerprintRebuildsFresh) {
    auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config);
        ASSERT_TRUE(manager.write_blocks(make_requests(42, make_pattern(11))));
    }

    config.model_fingerprint = "model-b";
    KVCacheOffloadManager manager(make_layout(), config);
    EXPECT_TRUE(manager.get_recovered_hashes().empty());
    EXPECT_EQ(manager.get_num_free_slots(), manager.get_num_slots());
}

TEST_F(PersistentCacheDirFixture, MismatchedTokenizerFingerprintRebuildsFresh) {
    auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config);
        ASSERT_TRUE(manager.write_blocks(make_requests(42, make_pattern(11))));
    }

    config.tokenizer_fingerprint = "tokenizer-b";
    KVCacheOffloadManager manager(make_layout(), config);
    EXPECT_TRUE(manager.get_recovered_hashes().empty());
}

TEST_F(PersistentCacheDirFixture, MismatchedLayoutRebuildsFresh) {
    auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config);
        ASSERT_TRUE(manager.write_blocks(make_requests(42, make_pattern(11))));
    }

    // A layout with a different key-segment size changes the layout fingerprint even though the
    // overall slot size (and therefore file sizes) stays comparable.
    KVCacheDiskLayout different_layout(std::vector<size_t>(NUM_LAYERS, KEY_BLOCK_BYTES + 8),
                                        std::vector<size_t>(NUM_LAYERS, VALUE_BLOCK_BYTES - 8));
    KVCacheOffloadManager manager(different_layout, config);
    EXPECT_TRUE(manager.get_recovered_hashes().empty());
}

TEST_F(PersistentCacheDirFixture, MismatchedSlotCountRebuildsFresh) {
    auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config);
        ASSERT_TRUE(manager.write_blocks(make_requests(42, make_pattern(11))));
    }

    config.capacity_bytes = 5 * SLOT_BYTES;
    KVCacheOffloadManager manager(make_layout(), config);
    EXPECT_TRUE(manager.get_recovered_hashes().empty());
    EXPECT_EQ(manager.get_num_slots(), 5u);
}

TEST_F(PersistentCacheDirFixture, TornManifestRecordIsIgnoredNotCrashed) {
    const auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config);
        ASSERT_TRUE(manager.write_blocks(make_requests(42, make_pattern(11))));
        ASSERT_TRUE(manager.write_blocks(make_requests(43, make_pattern(22))));
    }

    // Simulate a crash between the hash/checksum write and the valid-flag write for the second block's
    // slot: flip its valid flag back to 0 by directly patching the manifest file.
    constexpr std::size_t MANIFEST_HEADER_SIZE = 56;
    constexpr std::size_t MANIFEST_RECORD_SIZE = 24;
    const auto manifest_path = directory / "ov_genai_kv_offload.manifest";
    {
        std::fstream manifest(manifest_path, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(manifest.is_open());
        const std::size_t valid_flag_offset = MANIFEST_HEADER_SIZE + 1 * MANIFEST_RECORD_SIZE + 16;
        uint32_t zero = 0;
        manifest.seekp(static_cast<std::streamoff>(valid_flag_offset));
        manifest.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
    }

    KVCacheOffloadManager manager(make_layout(), config);
    const auto& recovered = manager.get_recovered_hashes();
    ASSERT_EQ(recovered.size(), 1u);
    EXPECT_EQ(recovered.front(), 42u);

    std::vector<uint8_t> actual(SLOT_BYTES);
    ASSERT_TRUE(manager.read_blocks(make_requests(42, actual)));
    EXPECT_EQ(actual, make_pattern(11));
}

TEST_F(PersistentCacheDirFixture, CorruptedDataDetectedAsChecksumMismatchOnRead) {
    const auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config);
        ASSERT_TRUE(manager.write_blocks(make_requests(42, make_pattern(11))));
    }

    // Corrupt the persisted data bytes for the first block directly on disk.
    const auto data_path = directory / "ov_genai_kv_offload.data";
    {
        std::fstream data_file(data_path, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(data_file.is_open());
        data_file.seekp(0);
        char garbage = '\xFF';
        data_file.write(&garbage, 1);
    }

    KVCacheOffloadManager manager(make_layout(), config);
    ASSERT_EQ(manager.get_recovered_hashes().size(), 1u);

    std::vector<uint8_t> actual(SLOT_BYTES);
    EXPECT_FALSE(manager.read_blocks(make_requests(42, actual)));
}

TEST_F(PersistentCacheDirFixture, RemovePersistedCacheDeletesFilesAndAllowsFreshStart) {
    const auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config);
        ASSERT_TRUE(manager.write_blocks(make_requests(42, make_pattern(11))));
    }

    ASSERT_TRUE(std::filesystem::exists(directory / "ov_genai_kv_offload.data"));
    ASSERT_TRUE(std::filesystem::exists(directory / "ov_genai_kv_offload.manifest"));

    KVCacheOffloadManager::remove_persisted_cache(directory);
    EXPECT_FALSE(std::filesystem::exists(directory / "ov_genai_kv_offload.data"));
    EXPECT_FALSE(std::filesystem::exists(directory / "ov_genai_kv_offload.manifest"));

    KVCacheOffloadManager manager(make_layout(), config);
    EXPECT_TRUE(manager.get_recovered_hashes().empty());
}

TEST_F(PersistentCacheDirFixture, MismatchedTenantIsolationSeedRebuildsFresh) {
    const auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config, /*tenant_isolation_seed=*/0);
        ASSERT_TRUE(manager.write_blocks(make_requests(42, make_pattern(11))));
    }

    // A different tenant/cache_salt combination must never recover a slot written under a different seed,
    // even though model, tokenizer, layout and slot count all still match.
    KVCacheOffloadManager manager(make_layout(), config,
                                  /*tenant_isolation_seed=*/compute_prefix_isolation_seed("tenant-a", ""));
    EXPECT_TRUE(manager.get_recovered_hashes().empty());
}

TEST_F(PersistentCacheDirFixture, SameTenantIsolationSeedRecoversAcrossRestart) {
    const auto config = make_persistent_config(3, directory);
    const uint64_t seed = compute_prefix_isolation_seed("tenant-a", "session-1");
    {
        KVCacheOffloadManager manager(make_layout(), config, seed);
        ASSERT_TRUE(manager.write_blocks(make_requests(42, make_pattern(11))));
    }

    KVCacheOffloadManager manager(make_layout(), config, seed);
    ASSERT_EQ(manager.get_recovered_hashes().size(), 1u);
    EXPECT_EQ(manager.get_recovered_hashes().front(), 42u);
}

TEST(TestKVCacheOffloadManager, RemovePersistedCacheOnEmptyDirectoryIsNoOp) {
    const auto directory = std::filesystem::temp_directory_path() / "ov_genai_offload_remove_noop_dir";
    std::filesystem::create_directories(directory);

    EXPECT_NO_THROW(KVCacheOffloadManager::remove_persisted_cache(directory));

    std::filesystem::remove_all(directory);
}

TEST(TestComputePrefixIsolationSeed, EmptyTenantAndSaltIsTheZeroSentinel) {
    EXPECT_EQ(compute_prefix_isolation_seed("", ""), 0u);
}

TEST(TestComputePrefixIsolationSeed, NonEmptyTenantIdIsNeverZero) {
    EXPECT_NE(compute_prefix_isolation_seed("tenant-a", ""), 0u);
}

TEST(TestComputePrefixIsolationSeed, NonEmptyCacheSaltAloneIsNeverZero) {
    EXPECT_NE(compute_prefix_isolation_seed("", "salt-a"), 0u);
}

TEST(TestComputePrefixIsolationSeed, IsDeterministic) {
    EXPECT_EQ(compute_prefix_isolation_seed("tenant-a", "salt-a"), compute_prefix_isolation_seed("tenant-a", "salt-a"));
}

TEST(TestComputePrefixIsolationSeed, DifferentTenantsProduceDifferentSeeds) {
    EXPECT_NE(compute_prefix_isolation_seed("tenant-a", ""), compute_prefix_isolation_seed("tenant-b", ""));
}

TEST(TestComputePrefixIsolationSeed, DifferentSaltsProduceDifferentSeedsForSameTenant) {
    EXPECT_NE(compute_prefix_isolation_seed("tenant-a", "salt-1"), compute_prefix_isolation_seed("tenant-a", "salt-2"));
}

TEST(TestComputePrefixIsolationSeed, ConcatenationAmbiguityIsNotCollapsed) {
    // Without an unambiguous separator, ("a", "bc") and ("ab", "c") would hash identically.
    EXPECT_NE(compute_prefix_isolation_seed("a", "bc"), compute_prefix_isolation_seed("ab", "c"));
}
