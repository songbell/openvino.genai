// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/i_kv_cache_storage_backend.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_offload_cache.hpp"
#include "continuous_batching/cache/kv_cache_offload_manager.hpp"
#include "helper.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/runtime/core.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace {

constexpr size_t NUM_DECODER_LAYERS = 2;
constexpr size_t NUM_PHYSICAL_BLOCKS = 4;

/// Owns the OpenVINO objects a KVCacheManager needs so that tests can allocate a real CPU KV cache.
struct CacheFixture {
    ov::Core core;
    ov::InferRequest request;
    std::shared_ptr<KVCacheManager> cache_manager;

    explicit CacheFixture(size_t num_blocks = NUM_PHYSICAL_BLOCKS) {
        request = core.compile_model(get_dummy_model(core, NUM_DECODER_LAYERS)).create_infer_request();
        cache_manager = std::make_shared<KVCacheManager>(request);
        cache_manager->allocate_cache_if_needed(num_blocks);
    }

    /// Fills a physical block with a byte pattern derived from `seed` and returns the expected bytes.
    std::vector<uint8_t> fill_block(size_t block_id, uint8_t seed) {
        std::vector<uint8_t> data(cache_manager->get_block_layout().get_slot_size());
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<uint8_t>(seed + i);
        }
        cache_manager->write_block(block_id, data);
        return data;
    }
};

CacheOffloadConfig make_offload_config(const CacheFixture& fixture, size_t num_slots) {
    CacheOffloadConfig config;
    config.capacity_bytes = fixture.cache_manager->get_block_layout().get_slot_size() * num_slots;
    return config;
}

std::unique_ptr<KVCacheOffloadCache> make_offload_cache(CacheFixture& fixture, size_t num_slots) {
    auto backend = std::make_unique<KVCacheOffloadManager>(fixture.cache_manager->get_block_layout(),
                                                           make_offload_config(fixture, num_slots),
                                                           "CPU");
    return std::make_unique<KVCacheOffloadCache>(*fixture.cache_manager, std::move(backend), /*max_queued_stores=*/8);
}

std::unique_ptr<KVCacheOffloadCache> make_offload_cache_with_host_pool(CacheFixture& fixture,
                                                                       size_t num_slots,
                                                                       size_t host_cache_slots) {
    auto backend = std::make_unique<KVCacheOffloadManager>(fixture.cache_manager->get_block_layout(),
                                                           make_offload_config(fixture, num_slots),
                                                           "CPU");
    return std::make_unique<KVCacheOffloadCache>(*fixture.cache_manager,
                                                 std::move(backend),
                                                 /*max_queued_stores=*/8,
                                                 /*wait_for_buffer=*/false,
                                                 /*enable_detailed_logging=*/false,
                                                 host_cache_slots);
}

BlocksPerLayer make_block_set(int physical_block_id) {
    return BlocksPerLayer{std::make_shared<CacheBlock>(physical_block_id)};
}

/// Minimal third-party-style IKVCacheStorageBackend: proves KVCacheOffloadCache's hash index, LRU
/// eviction and store/load orchestration live entirely on the KVCacheOffloadCache side of the seam, with
/// zero dependency on the file-based DefaultFileStorageBackend (KVCacheOffloadManager). Does not support
/// persistence (get_recovered_entries() is always empty), matching a plugin that declares no such
/// capability.
class InMemoryMockStorageBackend : public IKVCacheStorageBackend {
public:
    InMemoryMockStorageBackend(std::size_t slot_size, std::size_t num_slots)
        : m_slot_size(slot_size), m_slots(num_slots) {
        m_free_slots.reserve(num_slots);
        for (std::size_t i = num_slots; i-- > 0;) {
            m_free_slots.push_back(i);
        }
    }

    std::size_t get_slot_size() const override {
        return m_slot_size;
    }

    std::size_t get_num_slots() const override {
        return m_slots.size();
    }

    std::size_t get_num_free_slots() const override {
        return m_free_slots.size();
    }

    std::optional<std::size_t> acquire_slot() override {
        if (m_free_slots.empty()) {
            return std::nullopt;
        }
        const std::size_t slot_id = m_free_slots.back();
        m_free_slots.pop_back();
        return slot_id;
    }

    void release_slot(std::size_t slot_id) override {
        OPENVINO_ASSERT(slot_id < m_slots.size());
        m_free_slots.push_back(slot_id);
    }

    void write_slot(std::size_t slot_id,
                    const std::vector<uint8_t>& data,
                    std::optional<std::size_t> hash = std::nullopt) override {
        OPENVINO_ASSERT(slot_id < m_slots.size());
        OPENVINO_ASSERT(data.size() == m_slot_size);
        m_slots[slot_id] = data;
        ++num_writes;
        (void)hash;
    }

    void read_slot(std::size_t slot_id, std::vector<uint8_t>& data) const override {
        OPENVINO_ASSERT(slot_id < m_slots.size());
        data = m_slots[slot_id];
    }

    const std::vector<std::pair<std::size_t, std::size_t>>& get_recovered_entries() const override {
        static const std::vector<std::pair<std::size_t, std::size_t>> empty;
        return empty;
    }

    std::string describe() const override {
        return "InMemoryMockStorageBackend";
    }

    std::size_t num_writes = 0;

private:
    std::size_t m_slot_size;
    std::vector<std::vector<uint8_t>> m_slots;
    std::vector<std::size_t> m_free_slots;
};

std::unique_ptr<KVCacheOffloadCache> make_offload_cache_with_mock_backend(CacheFixture& fixture, size_t num_slots) {
    auto backend = std::make_unique<InMemoryMockStorageBackend>(fixture.cache_manager->get_block_layout().get_slot_size(),
                                                                 num_slots);
    return std::make_unique<KVCacheOffloadCache>(*fixture.cache_manager, std::move(backend), /*max_queued_stores=*/8);
}

SequenceGroup::Ptr make_group(const std::vector<int64_t>& tokens, uint64_t request_id) {
    return std::make_shared<SequenceGroup>(request_id,
                                           ov::Tensor(ov::element::i64, {tokens.size()}, const_cast<int64_t*>(tokens.data())),
                                           utils::get_greedy_config());
}

}  // namespace

TEST(TestKVCacheOffloadCache, HostCacheDisabledByDefaultKeepsNoHostEntries) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    fixture.fill_block(0, /*seed=*/1);

    offload_cache->on_blocks_overwritten(/*hash=*/9, make_block_set(0));
    offload_cache->flush();

    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    EXPECT_EQ(offload_cache->get_num_host_entries(), 0);
}

TEST(TestKVCacheOffloadCache, L1HitAvoidsDiskRead) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache_with_host_pool(fixture, /*num_slots=*/2, /*host_cache_slots=*/2);
    const auto expected = fixture.fill_block(0, /*seed=*/13);

    offload_cache->on_blocks_overwritten(/*hash=*/21, make_block_set(0));
    offload_cache->flush();
    ASSERT_EQ(offload_cache->get_num_host_entries(), 1);

    fixture.fill_block(1, /*seed=*/0);
    ASSERT_TRUE(offload_cache->load_into(/*hash=*/21, /*block_index=*/1));

    EXPECT_EQ(fixture.cache_manager->read_block(1), expected);
    EXPECT_EQ(offload_cache->get_statistics().num_load_host, 1);
    EXPECT_EQ(offload_cache->get_statistics().num_load_disk, 0);
}

TEST(TestKVCacheOffloadCache, HostPoolEvictsIndependentlyFromDisk) {
    CacheFixture fixture(/*num_blocks=*/NUM_PHYSICAL_BLOCKS);
    // Disk has room for both blocks, but the host pool only has room for the most recent one.
    auto offload_cache = make_offload_cache_with_host_pool(fixture, /*num_slots=*/4, /*host_cache_slots=*/1);
    const auto first = fixture.fill_block(0, /*seed=*/5);
    const auto second = fixture.fill_block(1, /*seed=*/90);

    offload_cache->on_blocks_overwritten(/*hash=*/100, make_block_set(0));
    offload_cache->flush();
    offload_cache->on_blocks_overwritten(/*hash=*/200, make_block_set(1));
    offload_cache->flush();

    // Both survive on disk...
    EXPECT_EQ(offload_cache->get_num_entries(), 2);
    // ...but only the newer one is still resident in the host pool.
    EXPECT_EQ(offload_cache->get_num_host_entries(), 1);
    EXPECT_EQ(offload_cache->get_statistics().num_host_evictions, 1);

    fixture.fill_block(2, /*seed=*/0);
    fixture.fill_block(3, /*seed=*/0);

    // The evicted-from-host hash still round-trips correctly, just via a disk read this time.
    ASSERT_TRUE(offload_cache->load_into(/*hash=*/100, /*block_index=*/2));
    EXPECT_EQ(fixture.cache_manager->read_block(2), first);
    EXPECT_EQ(offload_cache->get_statistics().num_load_disk, 1);

    ASSERT_TRUE(offload_cache->load_into(/*hash=*/200, /*block_index=*/3));
    EXPECT_EQ(fixture.cache_manager->read_block(3), second);
    EXPECT_EQ(offload_cache->get_statistics().num_load_host, 1);
}

TEST(TestKVCacheOffloadCache, GetLocationReportsUnknownHashAsNone) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);

    const auto location = offload_cache->get_location(/*hash=*/12345);
    EXPECT_FALSE(location.is_known());
    EXPECT_FALSE(location.in_flight);
    EXPECT_FALSE(location.host_resident);
    EXPECT_FALSE(location.disk_resident);
}

TEST(TestKVCacheOffloadCache, GetLocationNeverOverlapsInFlightWithDisk) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    fixture.fill_block(0, /*seed=*/4);

    offload_cache->on_blocks_overwritten(/*hash=*/55, make_block_set(0));
    // The background writer may already have published by now; either outcome is valid, but a hash must
    // never be observed as both still queued and already durable on disk at the same time.
    const auto location = offload_cache->get_location(55);
    EXPECT_TRUE(location.is_known());
    EXPECT_FALSE(location.in_flight && location.disk_resident);

    offload_cache->flush();
    const auto after_flush = offload_cache->get_location(55);
    EXPECT_FALSE(after_flush.in_flight);
    EXPECT_TRUE(after_flush.disk_resident);
}

TEST(TestKVCacheOffloadCache, GetLocationIsHostAndDiskAfterPublishWithHostCacheEnabled) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache_with_host_pool(fixture, /*num_slots=*/2, /*host_cache_slots=*/2);
    fixture.fill_block(0, /*seed=*/8);

    offload_cache->on_blocks_overwritten(/*hash=*/77, make_block_set(0));
    offload_cache->flush();

    const auto location = offload_cache->get_location(77);
    EXPECT_FALSE(location.in_flight);
    EXPECT_TRUE(location.host_resident);
    EXPECT_TRUE(location.disk_resident);
}

TEST(TestKVCacheOffloadCache, L1IsSeededBeforeDiskWriteCompletes) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache_with_host_pool(fixture, /*num_slots=*/2, /*host_cache_slots=*/2);
    const auto expected = fixture.fill_block(0, /*seed=*/17);

    offload_cache->on_blocks_overwritten(/*hash=*/88, make_block_set(0));
    // Host residency is set synchronously inside on_blocks_overwritten, so it must be visible immediately,
    // with no dependency on the background writer having reached disk yet.
    EXPECT_TRUE(offload_cache->get_location(88).host_resident);

    fixture.fill_block(1, /*seed=*/0);
    ASSERT_TRUE(offload_cache->load_into(/*hash=*/88, /*block_index=*/1));
    EXPECT_EQ(fixture.cache_manager->read_block(1), expected);
    // The background writer may already have published by the time load_into runs, so the hit can
    // legitimately land on either the in-flight or the host path; only their sum is deterministic.
    const auto stats = offload_cache->get_statistics();
    EXPECT_EQ(stats.num_load_staging + stats.num_load_host, 1);

    offload_cache->flush();
    EXPECT_TRUE(offload_cache->get_location(88).disk_resident);
}

TEST(TestKVCacheOffloadCache, GetLocationIsDiskOnlyWhenHostCacheDisabled) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);  // host_cache_slots defaults to 0
    fixture.fill_block(0, /*seed=*/2);

    offload_cache->on_blocks_overwritten(/*hash=*/33, make_block_set(0));
    offload_cache->flush();

    const auto location = offload_cache->get_location(33);
    EXPECT_FALSE(location.host_resident);
    EXPECT_TRUE(location.disk_resident);
}

TEST(TestKVCacheOffloadCache, GetLocationIsHostOnlyAfterDiskReclaimsSlot) {
    CacheFixture fixture(/*num_blocks=*/NUM_PHYSICAL_BLOCKS);
    // Disk has room for only one slot, the host pool has room for two: storing a second hash reclaims
    // the first hash's disk slot, but the host pool keeps both, so the first hash becomes host-only.
    auto offload_cache = make_offload_cache_with_host_pool(fixture, /*num_slots=*/1, /*host_cache_slots=*/2);
    fixture.fill_block(0, /*seed=*/6);
    fixture.fill_block(1, /*seed=*/70);

    offload_cache->on_blocks_overwritten(/*hash=*/300, make_block_set(0));
    offload_cache->flush();
    offload_cache->on_blocks_overwritten(/*hash=*/400, make_block_set(1));
    offload_cache->flush();

    const auto first_location = offload_cache->get_location(300);
    EXPECT_TRUE(first_location.host_resident);
    EXPECT_FALSE(first_location.disk_resident);

    const auto second_location = offload_cache->get_location(400);
    EXPECT_TRUE(second_location.host_resident);
    EXPECT_TRUE(second_location.disk_resident);
}

TEST(TestKVCacheOffloadCache, StoresBlockContentsOnOverwrite) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(1, /*seed=*/7);

    offload_cache->on_blocks_overwritten(/*hash=*/42, make_block_set(1));
    offload_cache->flush();

    EXPECT_TRUE(offload_cache->contains(42));
    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    EXPECT_EQ(offload_cache->get_statistics().num_stored, 1);
    EXPECT_EQ(offload_cache->get_statistics().num_queue_peak, 1);

    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(42, restored));
    EXPECT_EQ(restored, expected);
}

TEST(TestKVCacheOffloadCache, RejectsExcessiveBufferSlots) {
    CacheFixture fixture;
    auto backend = std::make_unique<KVCacheOffloadManager>(fixture.cache_manager->get_block_layout(),
                                                           make_offload_config(fixture, 1),
                                                           "CPU");

    EXPECT_THROW(KVCacheOffloadCache(*fixture.cache_manager,
                                     std::move(backend),
                                     CACHE_OFFLOAD_MAX_BUFFER_SLOTS + 1),
                 ov::Exception);
}

TEST(TestKVCacheOffloadCache, KeepsFirstCopyOfKnownHash) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(0, /*seed=*/3);
    offload_cache->on_blocks_overwritten(/*hash=*/5, make_block_set(0));

    fixture.fill_block(1, /*seed=*/200);
    offload_cache->on_blocks_overwritten(/*hash=*/5, make_block_set(1));
    offload_cache->flush();

    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    EXPECT_EQ(offload_cache->get_statistics().num_stored, 1);
    EXPECT_EQ(offload_cache->get_statistics().num_already_present, 1);

    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(5, restored));
    EXPECT_EQ(restored, expected);
}

TEST(TestKVCacheOffloadCache, KeepsDistinctHashesInSeparateSlots) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto first = fixture.fill_block(0, /*seed=*/11);
    const auto second = fixture.fill_block(1, /*seed=*/77);

    offload_cache->on_blocks_overwritten(/*hash=*/1, make_block_set(0));
    offload_cache->on_blocks_overwritten(/*hash=*/2, make_block_set(1));
    offload_cache->flush();

    ASSERT_EQ(offload_cache->get_num_entries(), 2);
    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(1, restored));
    EXPECT_EQ(restored, first);
    ASSERT_TRUE(offload_cache->read(2, restored));
    EXPECT_EQ(restored, second);
}

TEST(TestKVCacheOffloadCache, LoadIntoManyMatchesPerRequestOrderAndResults) {
    CacheFixture fixture(/*num_blocks=*/NUM_PHYSICAL_BLOCKS);
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto first = fixture.fill_block(0, /*seed=*/11);
    const auto second = fixture.fill_block(1, /*seed=*/77);

    offload_cache->on_blocks_overwritten(/*hash=*/1, make_block_set(0));
    offload_cache->on_blocks_overwritten(/*hash=*/2, make_block_set(1));
    offload_cache->flush();

    fixture.fill_block(2, /*seed=*/0);
    fixture.fill_block(3, /*seed=*/0);
    const auto results = offload_cache->load_into_many({{1, 2}, {2, 3}});

    ASSERT_EQ(results.size(), 2u);
    EXPECT_TRUE(results[0]);
    EXPECT_TRUE(results[1]);
    EXPECT_EQ(fixture.cache_manager->read_block(2), first);
    EXPECT_EQ(fixture.cache_manager->read_block(3), second);
    EXPECT_EQ(offload_cache->get_statistics().num_loaded, 2);
}

TEST(TestKVCacheOffloadCache, LoadIntoManyReportsPerRequestMissWithoutAffectingOthers) {
    CacheFixture fixture(/*num_blocks=*/NUM_PHYSICAL_BLOCKS);
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto known = fixture.fill_block(0, /*seed=*/9);

    offload_cache->on_blocks_overwritten(/*hash=*/1, make_block_set(0));
    offload_cache->flush();

    const auto untouched = fixture.fill_block(2, /*seed=*/0);
    fixture.fill_block(3, /*seed=*/0);
    // Middle request (hash 999) is unknown; it must not stop the other two from being served.
    const auto results = offload_cache->load_into_many({{1, 3}, {999, 2}, {1, 3}});

    ASSERT_EQ(results.size(), 3u);
    EXPECT_TRUE(results[0]);
    EXPECT_FALSE(results[1]);
    EXPECT_TRUE(results[2]);
    EXPECT_EQ(fixture.cache_manager->read_block(3), known);
    // block_index=2 was never touched since its request (hash 999) missed.
    EXPECT_EQ(fixture.cache_manager->read_block(2), untouched);
    EXPECT_EQ(offload_cache->get_statistics().num_load_misses, 1);
}

TEST(TestKVCacheOffloadCache, ReplacesOldestEntryWhenFileIsFull) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    fixture.fill_block(0, /*seed=*/1);
    offload_cache->on_blocks_overwritten(/*hash=*/100, make_block_set(0));
    offload_cache->flush();
    ASSERT_EQ(offload_cache->get_num_free_slots(), 0);

    const auto newer = fixture.fill_block(1, /*seed=*/60);
    offload_cache->on_blocks_overwritten(/*hash=*/200, make_block_set(1));
    offload_cache->flush();

    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    EXPECT_FALSE(offload_cache->contains(100));
    EXPECT_EQ(offload_cache->get_statistics().num_replaced, 1);

    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(200, restored));
    EXPECT_EQ(restored, newer);
}

TEST(TestKVCacheOffloadCache, ReadReportsMissForUnknownHash) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);

    std::vector<uint8_t> restored;
    EXPECT_FALSE(offload_cache->read(/*hash=*/999, restored));
    EXPECT_TRUE(restored.empty());
}

TEST(TestKVCacheOffloadCache, RejectsPerLayerBlockTables) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    BlocksPerLayer per_layer_blocks{std::make_shared<CacheBlock>(0), std::make_shared<CacheBlock>(1)};

    EXPECT_THROW(offload_cache->on_blocks_overwritten(/*hash=*/1, per_layer_blocks), ov::Exception);
}

TEST(TestKVCacheOffloadCache, RemovesOffloadFileOnDestruction) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    fixture.fill_block(0, /*seed=*/5);
    offload_cache->on_blocks_overwritten(/*hash=*/1, make_block_set(0));

    offload_cache.reset();
    // The backend owns a run-specific file that must not outlive the cache.
    EXPECT_EQ(fixture.cache_manager->get_num_allocated_blocks(), NUM_PHYSICAL_BLOCKS);
}

TEST(TestKVCacheOffloadCache, BlockManagerStoresEvictedPrefixBlock) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/2);
    auto offload_cache = make_offload_cache(fixture, 2);

    BlockManager block_manager(/*num_blocks=*/2, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    block_manager.set_overwritten_block_observer(offload_cache.get());

    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto producer = make_group(tokens, /*request_id=*/1);
    producer->schedule_tokens(tokens.size());
    block_manager.append_slots(producer);
    producer->finish_iteration();

    const auto producer_seq_id = producer->get_running_sequences().at(0)->get_id();
    ASSERT_EQ(block_manager.get_block_table(producer_seq_id, 0).size(), 2);
    block_manager.free_sequence(producer_seq_id);
    ASSERT_EQ(offload_cache->get_num_entries(), 0);

    // Allocating for an unrelated prefix exhausts the free pool and overwrites a cached block.
    const std::vector<int64_t> pressure_tokens = {10, 11, 12, 13, 14, 15, 16, 17};
    auto pressure = make_group(pressure_tokens, /*request_id=*/2);
    pressure->schedule_tokens(pressure_tokens.size());
    block_manager.append_slots(pressure);
    offload_cache->flush();

    EXPECT_GT(offload_cache->get_num_entries(), 0);
    EXPECT_EQ(offload_cache->get_statistics().num_stored, offload_cache->get_num_entries());

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(pressure->get_running_sequences().at(0)->get_id());
}

TEST(TestKVCacheOffloadCache, DetachedObserverStopsStoring) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/2);
    auto offload_cache = make_offload_cache(fixture, 2);

    BlockManager block_manager(/*num_blocks=*/2, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    block_manager.set_overwritten_block_observer(offload_cache.get());
    block_manager.set_overwritten_block_observer(nullptr);

    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto producer = make_group(tokens, /*request_id=*/3);
    producer->schedule_tokens(tokens.size());
    block_manager.append_slots(producer);
    producer->finish_iteration();
    block_manager.free_sequence(producer->get_running_sequences().at(0)->get_id());

    const std::vector<int64_t> pressure_tokens = {20, 21, 22, 23, 24, 25, 26, 27};
    auto pressure = make_group(pressure_tokens, /*request_id=*/4);
    pressure->schedule_tokens(pressure_tokens.size());
    block_manager.append_slots(pressure);
    offload_cache->flush();

    EXPECT_EQ(offload_cache->get_num_entries(), 0);

    block_manager.free_sequence(pressure->get_running_sequences().at(0)->get_id());
}

TEST(TestKVCacheOffloadCache, LoadsStoredContentsIntoAnotherBlock) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(0, /*seed=*/23);
    offload_cache->on_blocks_overwritten(/*hash=*/77, make_block_set(0));
    offload_cache->flush();
    fixture.fill_block(2, /*seed=*/0);

    ASSERT_TRUE(offload_cache->load_into(/*hash=*/77, /*block_index=*/2));

    EXPECT_EQ(fixture.cache_manager->read_block(2), expected);
    EXPECT_EQ(offload_cache->get_statistics().num_loaded, 1);
    EXPECT_EQ(offload_cache->get_statistics().num_load_disk, 1);
}

TEST(TestKVCacheOffloadCache, LoadReportsMissForUnknownHash) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    const auto untouched = fixture.fill_block(1, /*seed=*/9);

    EXPECT_FALSE(offload_cache->load_into(/*hash=*/12345, /*block_index=*/1));
    EXPECT_EQ(fixture.cache_manager->read_block(1), untouched);
    EXPECT_EQ(offload_cache->get_statistics().num_load_misses, 1);
}

TEST(TestKVCacheOffloadCache, WarmPrefixCacheRestoresBlocksFromDisk) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/2);
    auto offload_cache = make_offload_cache(fixture, 2);

    BlockManager block_manager(/*num_blocks=*/2, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    block_manager.set_overwritten_block_observer(offload_cache.get());

    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto producer = make_group(tokens, /*request_id=*/5);
    producer->schedule_tokens(tokens.size());
    block_manager.append_slots(producer);
    producer->finish_iteration();
    block_manager.free_sequence(producer->get_running_sequences().at(0)->get_id());

    // Evict the cached prefix from memory, which pushes its contents to disk.
    const std::vector<int64_t> pressure_tokens = {30, 31, 32, 33, 34, 35, 36, 37};
    auto pressure = make_group(pressure_tokens, /*request_id=*/6);
    pressure->schedule_tokens(pressure_tokens.size());
    block_manager.append_slots(pressure);
    const auto pressure_seq_id = pressure->get_running_sequences().at(0)->get_id();
    offload_cache->flush();
    ASSERT_GT(offload_cache->get_num_entries(), 0);
    block_manager.free_sequence(pressure_seq_id);

    auto consumer = make_group(tokens, /*request_id=*/7);
    size_t num_warmed = 0;
    {
        KVCacheOffloadCache::ScopedReclamationPause keep_entries(*offload_cache);
        num_warmed = block_manager.warm_prefix_cache(consumer, *offload_cache);
    }

    EXPECT_GT(num_warmed, 0);
    EXPECT_EQ(offload_cache->get_statistics().num_loaded, num_warmed);
    // The warmed blocks are unowned, so the regular restore path can now claim them.
    EXPECT_TRUE(block_manager.restore_cached_blocks(consumer));
    EXPECT_GT(consumer->get_num_processed_tokens(), 0);

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(consumer->get_running_sequences().at(0)->get_id());
}

TEST(TestKVCacheOffloadCache, WarmPrefixCacheIsNoOpWithoutEntries) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/2);
    auto offload_cache = make_offload_cache(fixture, 2);

    BlockManager block_manager(/*num_blocks=*/2, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto consumer = make_group(tokens, /*request_id=*/8);

    EXPECT_EQ(block_manager.warm_prefix_cache(consumer, *offload_cache), 0);
    EXPECT_EQ(block_manager.num_free_blocks(), 2);
}

TEST(TestKVCacheOffloadCache, WarmPrefixCacheKeepsChainWhenNoBlockIsUnused) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/2);
    auto offload_cache = make_offload_cache(fixture, 4);

    BlockManager block_manager(/*num_blocks=*/2, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    block_manager.set_overwritten_block_observer(offload_cache.get());

    const std::vector<int64_t> wanted_tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto producer = make_group(wanted_tokens, /*request_id=*/9);
    producer->schedule_tokens(wanted_tokens.size());
    block_manager.append_slots(producer);
    producer->finish_iteration();
    block_manager.free_sequence(producer->get_running_sequences().at(0)->get_id());

    // Push both blocks of the wanted prefix to disk and leave the cache full of unrelated blocks.
    const std::vector<int64_t> other_tokens = {40, 41, 42, 43, 44, 45, 46, 47};
    auto other = make_group(other_tokens, /*request_id=*/10);
    other->schedule_tokens(other_tokens.size());
    block_manager.append_slots(other);
    other->finish_iteration();
    block_manager.free_sequence(other->get_running_sequences().at(0)->get_id());
    offload_cache->flush();
    ASSERT_EQ(offload_cache->get_num_entries(), 2);

    auto consumer = make_group(wanted_tokens, /*request_id=*/11);
    // Both chain blocks have to survive warming even though every block has to be taken from the cache.
    {
        KVCacheOffloadCache::ScopedReclamationPause keep_entries(*offload_cache);
        EXPECT_EQ(block_manager.warm_prefix_cache(consumer, *offload_cache), 2);
    }
    ASSERT_TRUE(block_manager.restore_cached_blocks(consumer));

    const auto consumer_seq_id = consumer->get_running_sequences().at(0)->get_id();
    EXPECT_EQ(block_manager.get_block_table(consumer_seq_id, 0).size(), 2);

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(consumer_seq_id);
}

TEST(TestKVCacheOffloadCache, StoresBlocksFreedByPartialRelease) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/4);
    auto offload_cache = make_offload_cache(fixture, 4);

    BlockManager block_manager(/*num_blocks=*/4, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    block_manager.set_overwritten_block_observer(offload_cache.get());

    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto preempted = make_group(tokens, /*request_id=*/12);
    preempted->schedule_tokens(tokens.size());
    block_manager.append_slots(preempted);
    preempted->finish_iteration();

    const auto seq_id = preempted->get_running_sequences().at(0)->get_id();
    ASSERT_EQ(block_manager.get_block_table(seq_id, 0).size(), 2);

    // Preemption releases blocks from the tail of the sequence.
    block_manager.free_sequence_partially(seq_id, 1);
    EXPECT_EQ(block_manager.get_block_table(seq_id, 0).size(), 1);

    // Fill the cache so the released block is overwritten and therefore offloaded.
    const std::vector<int64_t> other_tokens = {50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61};
    auto other = make_group(other_tokens, /*request_id=*/13);
    other->schedule_tokens(other_tokens.size());
    block_manager.append_slots(other);
    offload_cache->flush();

    EXPECT_GT(offload_cache->get_num_entries(), 0);

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(seq_id);
    block_manager.free_sequence(other->get_running_sequences().at(0)->get_id());
}

TEST(TestKVCacheOffloadCache, ServesStoredBlockWhileTheWriteIsStillQueued) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(0, /*seed=*/31);

    // The store hands off to a background writer, yet the contents must be readable right away.
    offload_cache->on_blocks_overwritten(/*hash=*/64, make_block_set(0));
    EXPECT_TRUE(offload_cache->contains(64));

    fixture.fill_block(3, /*seed=*/0);
    ASSERT_TRUE(offload_cache->load_into(/*hash=*/64, /*block_index=*/3));
    EXPECT_EQ(fixture.cache_manager->read_block(3), expected);
    // The background writer may have already published by the time load_into runs, so the hit can
    // legitimately land on either source; only their sum is deterministic.
    EXPECT_EQ(offload_cache->get_statistics().num_load_staging + offload_cache->get_statistics().num_load_disk, 1);

    offload_cache->flush();
    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(64, restored));
    EXPECT_EQ(restored, expected);
}

TEST(TestKVCacheOffloadCache, HandlesCopyOnWriteOfSharedBlock) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/8);
    auto offload_cache = make_offload_cache(fixture, 8);

    BlockManager block_manager(/*num_blocks=*/8, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    block_manager.set_overwritten_block_observer(offload_cache.get());

    // The trailing block stays incomplete, so consumers sharing it have to split it on the next token.
    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5};
    auto producer = make_group(tokens, /*request_id=*/14);
    producer->schedule_tokens(tokens.size());
    block_manager.append_slots(producer);
    producer->finish_iteration();
    const auto producer_seq_id = producer->get_running_sequences().at(0)->get_id();
    const auto shared_block_idx = block_manager.get_block_table(producer_seq_id, 0).at(1)->get_index();
    block_manager.free_sequence(producer_seq_id);

    auto first = make_group(tokens, /*request_id=*/15);
    auto second = make_group(tokens, /*request_id=*/16);
    ASSERT_TRUE(block_manager.restore_cached_blocks(first));
    ASSERT_TRUE(block_manager.restore_cached_blocks(second));

    first->schedule_tokens(1);
    second->schedule_tokens(1);
    const auto first_copy_map = block_manager.append_slots(first);
    const auto second_copy_map = block_manager.append_slots(second);

    EXPECT_TRUE(first_copy_map.count(shared_block_idx));
    EXPECT_TRUE(second_copy_map.empty());
    // Copy-on-write must not publish anything to disk, since no cached contents were overwritten.
    offload_cache->flush();
    EXPECT_EQ(offload_cache->get_num_entries(), 0);

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(first->get_running_sequences().at(0)->get_id());
    block_manager.free_sequence(second->get_running_sequences().at(0)->get_id());
}

TEST(TestKVCacheOffloadCache, RecoversEntriesFromPersistedCacheAcrossRestart) {
    const auto directory = std::filesystem::temp_directory_path() /
                           ("ov_genai_offload_cache_persist_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
    std::filesystem::create_directories(directory);

    CacheFixture fixture;
    CacheOffloadConfig config = make_offload_config(fixture, /*num_slots=*/2);
    config.path = directory.string();
    config.enable_persistence = true;
    config.model_fingerprint = "model-x";
    config.tokenizer_fingerprint = "tokenizer-x";

    const auto expected = fixture.fill_block(0, /*seed=*/3);
    {
        auto backend = std::make_unique<KVCacheOffloadManager>(fixture.cache_manager->get_block_layout(), config, "CPU");
        KVCacheOffloadCache offload_cache(*fixture.cache_manager, std::move(backend), /*max_queued_stores=*/8);
        offload_cache.on_blocks_overwritten(/*hash=*/21, make_block_set(0));
        offload_cache.flush();
        EXPECT_EQ(offload_cache.get_num_entries(), 1);
        // offload_cache and its backend destruct here; the persisted files must survive.
    }

    auto reopened_backend = std::make_unique<KVCacheOffloadManager>(fixture.cache_manager->get_block_layout(), config, "CPU");
    ASSERT_EQ(reopened_backend->get_recovered_entries().size(), 1u);
    {
        KVCacheOffloadCache reopened_cache(*fixture.cache_manager, std::move(reopened_backend), /*max_queued_stores=*/8);

        EXPECT_EQ(reopened_cache.get_num_entries(), 1u);
        const auto location = reopened_cache.get_location(/*hash=*/21);
        EXPECT_TRUE(location.disk_resident);

        std::vector<uint8_t> actual;
        EXPECT_TRUE(reopened_cache.read(/*hash=*/21, actual));
        EXPECT_EQ(actual, expected);
    }

    std::filesystem::remove_all(directory);
}

TEST(TestKVCacheOffloadCache, WorksWithThirdPartyStorageBackend) {
    // Proves the IKVCacheStorageBackend seam is real: KVCacheOffloadCache's hash index, publish and load
    // orchestration behave identically when the backend is a non-file, in-memory mock rather than the
    // built-in DefaultFileStorageBackend (KVCacheOffloadManager).
    CacheFixture fixture;
    auto offload_cache = make_offload_cache_with_mock_backend(fixture, /*num_slots=*/2);
    const auto expected = fixture.fill_block(0, /*seed=*/9);

    offload_cache->on_blocks_overwritten(/*hash=*/55, make_block_set(0));
    offload_cache->flush();

    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    EXPECT_TRUE(offload_cache->contains(55));

    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(/*hash=*/55, restored));
    EXPECT_EQ(restored, expected);

    EXPECT_TRUE(offload_cache->load_into(55, /*block_index=*/1));
    const auto loaded = fixture.cache_manager->read_block(1);
    EXPECT_EQ(loaded, expected);
}

TEST(TestKVCacheOffloadCache, EvictionAndMissPolicyAreBackendAgnostic) {
    // The mock backend has no eviction logic of its own (release_slot() only returns the slot to a free
    // list); LRU replacement and miss reporting must therefore be entirely KVCacheOffloadCache's doing.
    CacheFixture fixture;
    auto offload_cache = make_offload_cache_with_mock_backend(fixture, /*num_slots=*/1);
    fixture.fill_block(0, /*seed=*/1);
    offload_cache->on_blocks_overwritten(/*hash=*/100, make_block_set(0));
    offload_cache->flush();
    ASSERT_EQ(offload_cache->get_num_free_slots(), 0);

    const auto newer = fixture.fill_block(1, /*seed=*/60);
    offload_cache->on_blocks_overwritten(/*hash=*/200, make_block_set(1));
    offload_cache->flush();

    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    EXPECT_FALSE(offload_cache->contains(100));
    EXPECT_EQ(offload_cache->get_statistics().num_replaced, 1);

    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(200, restored));
    EXPECT_EQ(restored, newer);

    std::vector<uint8_t> miss;
    EXPECT_FALSE(offload_cache->read(/*hash=*/999, miss));
}
