// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

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

}  // namespace

TEST(TestKVCacheOffloadManager, DerivesSlotCountFromCapacity) {
    // Capacity that is not a multiple of the slot size must be rounded down.
    KVCacheOffloadManager manager(make_layout(), make_config(4), "CPU");

    EXPECT_EQ(manager.get_slot_size(), SLOT_BYTES);
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
        KVCacheOffloadManager manager(make_layout(), make_config(2), "CPU");
        file_path = manager.get_file_path();

        ASSERT_FALSE(file_path.empty());
        EXPECT_TRUE(std::filesystem::exists(file_path));
        EXPECT_EQ(std::filesystem::file_size(file_path), 2 * SLOT_BYTES);
    }
    EXPECT_FALSE(std::filesystem::exists(file_path));
}

TEST(TestKVCacheOffloadManager, SlotRoundTrip) {
    KVCacheOffloadManager manager(make_layout(), make_config(2), "CPU");
    const auto expected = make_pattern(7);

    manager.write_slot(1, expected);

    std::vector<uint8_t> actual;
    manager.read_slot(1, actual);
    EXPECT_EQ(actual, expected);
}

TEST(TestKVCacheOffloadManager, SlotsDoNotOverlap) {
    KVCacheOffloadManager manager(make_layout(), make_config(3), "CPU");
    const auto first = make_pattern(1);
    const auto second = make_pattern(200);

    manager.write_slot(0, first);
    manager.write_slot(2, second);

    std::vector<uint8_t> actual;
    manager.read_slot(0, actual);
    EXPECT_EQ(actual, first);
    manager.read_slot(2, actual);
    EXPECT_EQ(actual, second);
}

TEST(TestKVCacheOffloadManager, AcquireReleaseAndExhaustion) {
    KVCacheOffloadManager manager(make_layout(), make_config(2), "CPU");

    const auto first = manager.acquire_slot();
    const auto second = manager.acquire_slot();
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_NE(*first, *second);
    EXPECT_EQ(manager.get_num_free_slots(), 0);

    EXPECT_FALSE(manager.acquire_slot().has_value());

    manager.release_slot(*first);
    EXPECT_EQ(manager.get_num_free_slots(), 1);
    EXPECT_EQ(manager.acquire_slot(), first);
}

TEST(TestKVCacheOffloadManager, RejectsDoubleRelease) {
    KVCacheOffloadManager manager(make_layout(), make_config(2), "CPU");
    const auto slot = manager.acquire_slot();
    ASSERT_TRUE(slot.has_value());

    manager.release_slot(*slot);
    EXPECT_THROW(manager.release_slot(*slot), ov::Exception);
}

TEST(TestKVCacheOffloadManager, RejectsInvalidSlotAndBlockSize) {
    KVCacheOffloadManager manager(make_layout(), make_config(2), "CPU");
    std::vector<uint8_t> block_data;

    EXPECT_THROW(manager.write_slot(2, make_pattern(0)), ov::Exception);
    EXPECT_THROW(manager.read_slot(2, block_data), ov::Exception);
    EXPECT_THROW(manager.release_slot(2), ov::Exception);

    std::vector<uint8_t> too_short(SLOT_BYTES - 1, 0);
    EXPECT_THROW(manager.write_slot(0, too_short), ov::Exception);
}

TEST(TestKVCacheOffloadManager, RejectsCapacitySmallerThanOneBlock) {
    CacheOffloadConfig config;
    config.capacity_bytes = SLOT_BYTES - 1;

    EXPECT_THROW(KVCacheOffloadManager(make_layout(), config, "CPU"), ov::Exception);
}

TEST(TestKVCacheOffloadManager, RejectsUnsupportedDeviceAndDirectIO) {
    EXPECT_TRUE(KVCacheOffloadManager::is_supported_device("CPU"));
    EXPECT_TRUE(KVCacheOffloadManager::is_supported_device("GPU"));
    EXPECT_TRUE(KVCacheOffloadManager::is_supported_device("GPU.0"));
    EXPECT_FALSE(KVCacheOffloadManager::is_supported_device("NPU"));

    EXPECT_THROW(KVCacheOffloadManager(make_layout(), make_config(1), "NPU"), ov::Exception);

    CacheOffloadConfig direct_io = make_config(1);
    direct_io.use_page_cache = false;
    EXPECT_THROW(KVCacheOffloadManager(make_layout(), direct_io, "CPU"), ov::Exception);
}

TEST(TestKVCacheOffloadManager, RejectsMissingDirectory) {
    CacheOffloadConfig config = make_config(1);
    config.path = (std::filesystem::temp_directory_path() / "ov_genai_offload_missing_dir").string();
    ASSERT_FALSE(std::filesystem::exists(config.path));

    EXPECT_THROW(KVCacheOffloadManager(make_layout(), config, "CPU"), ov::Exception);
}

TEST(TestKVCacheOffloadManager, UsesConfiguredDirectory) {
    const auto directory = std::filesystem::temp_directory_path() / "ov_genai_offload_test_dir";
    std::filesystem::create_directories(directory);

    CacheOffloadConfig config = make_config(1);
    config.path = directory.string();

    {
        KVCacheOffloadManager manager(make_layout(), config, "CPU");
        EXPECT_EQ(manager.get_file_path().parent_path(), directory);
    }

    std::filesystem::remove_all(directory);
}

namespace {

CacheOffloadConfig make_persistent_config(size_t num_slots, const std::filesystem::path& directory) {
    CacheOffloadConfig config = make_config(num_slots);
    config.path = directory.string();
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
    EXPECT_THROW(KVCacheOffloadManager(make_layout(), config, "CPU"), ov::Exception);

    config.path = directory.string();
    // Fingerprints still empty.
    EXPECT_THROW(KVCacheOffloadManager(make_layout(), config, "CPU"), ov::Exception);

    config.model_fingerprint = "model-a";
    // tokenizer_fingerprint still empty.
    EXPECT_THROW(KVCacheOffloadManager(make_layout(), config, "CPU"), ov::Exception);
}

TEST_F(PersistentCacheDirFixture, FreshPersistentCacheHasNoRecoveredEntries) {
    KVCacheOffloadManager manager(make_layout(), make_persistent_config(3, directory), "CPU");
    EXPECT_TRUE(manager.get_recovered_entries().empty());
    EXPECT_TRUE(std::filesystem::exists(manager.get_file_path()));
}

TEST_F(PersistentCacheDirFixture, SurvivesDestructionAndIsRecoveredOnReopen) {
    const auto config = make_persistent_config(3, directory);
    const auto first_data = make_pattern(11);
    const auto second_data = make_pattern(99);

    {
        KVCacheOffloadManager manager(make_layout(), config, "CPU");
        manager.write_slot(0, first_data, /*hash=*/42);
        manager.write_slot(2, second_data, /*hash=*/77);
        // manager destructs here; files must survive since persistence is enabled.
    }

    ASSERT_TRUE(std::filesystem::exists(directory / "ov_genai_kv_offload.data"));
    ASSERT_TRUE(std::filesystem::exists(directory / "ov_genai_kv_offload.manifest"));

    KVCacheOffloadManager manager(make_layout(), config, "CPU");
    const auto& recovered = manager.get_recovered_entries();
    ASSERT_EQ(recovered.size(), 2u);

    std::size_t slot_for_42 = static_cast<std::size_t>(-1);
    std::size_t slot_for_77 = static_cast<std::size_t>(-1);
    for (const auto& entry : recovered) {
        if (entry.first == 42) {
            slot_for_42 = entry.second;
        } else if (entry.first == 77) {
            slot_for_77 = entry.second;
        }
    }
    ASSERT_EQ(slot_for_42, 0u);
    ASSERT_EQ(slot_for_77, 2u);

    std::vector<uint8_t> actual;
    manager.read_slot(0, actual);
    EXPECT_EQ(actual, first_data);
    manager.read_slot(2, actual);
    EXPECT_EQ(actual, second_data);

    // Recovered slots must not be handed out as free slots.
    EXPECT_EQ(manager.get_num_free_slots(), 1u);
}

TEST_F(PersistentCacheDirFixture, MismatchedModelFingerprintRebuildsFresh) {
    auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config, "CPU");
        manager.write_slot(0, make_pattern(11), /*hash=*/42);
    }

    config.model_fingerprint = "model-b";
    KVCacheOffloadManager manager(make_layout(), config, "CPU");
    EXPECT_TRUE(manager.get_recovered_entries().empty());
    EXPECT_EQ(manager.get_num_free_slots(), manager.get_num_slots());
}

TEST_F(PersistentCacheDirFixture, MismatchedTokenizerFingerprintRebuildsFresh) {
    auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config, "CPU");
        manager.write_slot(0, make_pattern(11), /*hash=*/42);
    }

    config.tokenizer_fingerprint = "tokenizer-b";
    KVCacheOffloadManager manager(make_layout(), config, "CPU");
    EXPECT_TRUE(manager.get_recovered_entries().empty());
}

TEST_F(PersistentCacheDirFixture, MismatchedLayoutRebuildsFresh) {
    auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config, "CPU");
        manager.write_slot(0, make_pattern(11), /*hash=*/42);
    }

    // A layout with a different key-segment size changes the layout fingerprint even though the
    // overall slot size (and therefore file sizes) stays comparable.
    KVCacheDiskLayout different_layout(std::vector<size_t>(NUM_LAYERS, KEY_BLOCK_BYTES + 8),
                                        std::vector<size_t>(NUM_LAYERS, VALUE_BLOCK_BYTES - 8));
    KVCacheOffloadManager manager(different_layout, config, "CPU");
    EXPECT_TRUE(manager.get_recovered_entries().empty());
}

TEST_F(PersistentCacheDirFixture, MismatchedSlotCountRebuildsFresh) {
    auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config, "CPU");
        manager.write_slot(0, make_pattern(11), /*hash=*/42);
    }

    config.capacity_bytes = 5 * SLOT_BYTES;
    KVCacheOffloadManager manager(make_layout(), config, "CPU");
    EXPECT_TRUE(manager.get_recovered_entries().empty());
    EXPECT_EQ(manager.get_num_slots(), 5u);
}

TEST_F(PersistentCacheDirFixture, TornManifestRecordIsIgnoredNotCrashed) {
    const auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config, "CPU");
        manager.write_slot(0, make_pattern(11), /*hash=*/42);
        manager.write_slot(1, make_pattern(22), /*hash=*/43);
    }

    // Simulate a crash between the hash/checksum write and the valid-flag write for slot 1: flip its
    // valid flag back to 0 by directly patching the manifest file.
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

    KVCacheOffloadManager manager(make_layout(), config, "CPU");
    const auto& recovered = manager.get_recovered_entries();
    ASSERT_EQ(recovered.size(), 1u);
    EXPECT_EQ(recovered.front().first, 42u);
    EXPECT_EQ(recovered.front().second, 0u);

    std::vector<uint8_t> actual;
    manager.read_slot(0, actual);
    EXPECT_EQ(actual, make_pattern(11));
}

TEST_F(PersistentCacheDirFixture, CorruptedDataDetectedAsChecksumMismatchOnRead) {
    const auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config, "CPU");
        manager.write_slot(0, make_pattern(11), /*hash=*/42);
    }

    // Corrupt the persisted data bytes for slot 0 directly on disk.
    const auto data_path = directory / "ov_genai_kv_offload.data";
    {
        std::fstream data_file(data_path, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(data_file.is_open());
        data_file.seekp(0);
        char garbage = '\xFF';
        data_file.write(&garbage, 1);
    }

    KVCacheOffloadManager manager(make_layout(), config, "CPU");
    ASSERT_EQ(manager.get_recovered_entries().size(), 1u);

    std::vector<uint8_t> actual;
    EXPECT_THROW(manager.read_slot(0, actual), ov::Exception);
}

TEST_F(PersistentCacheDirFixture, WriteWithoutHashIsNotPersisted) {
    const auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config, "CPU");
        manager.write_slot(0, make_pattern(11));  // no hash: soft degradation, not an error.
    }

    KVCacheOffloadManager manager(make_layout(), config, "CPU");
    EXPECT_TRUE(manager.get_recovered_entries().empty());
}

TEST_F(PersistentCacheDirFixture, RemovePersistedCacheDeletesFilesAndAllowsFreshStart) {
    const auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config, "CPU");
        manager.write_slot(0, make_pattern(11), /*hash=*/42);
    }

    ASSERT_TRUE(std::filesystem::exists(directory / "ov_genai_kv_offload.data"));
    ASSERT_TRUE(std::filesystem::exists(directory / "ov_genai_kv_offload.manifest"));

    KVCacheOffloadManager::remove_persisted_cache(directory);
    EXPECT_FALSE(std::filesystem::exists(directory / "ov_genai_kv_offload.data"));
    EXPECT_FALSE(std::filesystem::exists(directory / "ov_genai_kv_offload.manifest"));

    KVCacheOffloadManager manager(make_layout(), config, "CPU");
    EXPECT_TRUE(manager.get_recovered_entries().empty());
}

TEST_F(PersistentCacheDirFixture, MismatchedTenantIsolationSeedRebuildsFresh) {
    const auto config = make_persistent_config(3, directory);
    {
        KVCacheOffloadManager manager(make_layout(), config, "CPU", /*tenant_isolation_seed=*/0);
        manager.write_slot(0, make_pattern(11), /*hash=*/42);
    }

    // A different tenant/cache_salt combination must never recover a slot written under a different seed,
    // even though model, tokenizer, layout and slot count all still match.
    KVCacheOffloadManager manager(make_layout(), config, "CPU",
                                  /*tenant_isolation_seed=*/compute_prefix_isolation_seed("tenant-a", ""));
    EXPECT_TRUE(manager.get_recovered_entries().empty());
}

TEST_F(PersistentCacheDirFixture, SameTenantIsolationSeedRecoversAcrossRestart) {
    const auto config = make_persistent_config(3, directory);
    const uint64_t seed = compute_prefix_isolation_seed("tenant-a", "session-1");
    {
        KVCacheOffloadManager manager(make_layout(), config, "CPU", seed);
        manager.write_slot(0, make_pattern(11), /*hash=*/42);
    }

    KVCacheOffloadManager manager(make_layout(), config, "CPU", seed);
    ASSERT_EQ(manager.get_recovered_entries().size(), 1u);
    EXPECT_EQ(manager.get_recovered_entries().front().first, 42u);
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
