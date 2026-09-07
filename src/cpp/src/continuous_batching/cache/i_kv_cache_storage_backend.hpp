// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ov::genai {

/**
 * @brief Boundary between the KV cache offload cache (hash -> slot index, publish/load orchestration,
 * eviction policy) and a slot storage backend (slot -> bytes).
 *
 * A backend owns a fixed number of fixed-size slots and moves whole-slot byte buffers in and out; it is
 * deliberately kept ignorant of block hashes, sequences, `RemoteTensor`, or any prefix-cache policy - all
 * of that stays in `KVCacheOffloadCache`/`BlockManager`. This is the seam a third-party storage plugin
 * would sit behind (see `kv_cache_offload_new_plan.md` Phase 6): `KVCacheOffloadCache` only ever talks to
 * this interface, never to a concrete backend type.
 *
 * `DefaultFileStorageBackend` (implemented by `KVCacheOffloadManager`) is the only implementation today.
 */
class IKVCacheStorageBackend {
public:
    virtual ~IKVCacheStorageBackend() = default;

    /// @return Size of one slot in bytes, equal to the byte size of one block set across all layers.
    virtual std::size_t get_slot_size() const = 0;

    /// @return Total number of slots this backend was configured with.
    virtual std::size_t get_num_slots() const = 0;

    /// @return Number of slots not currently held by acquire_slot().
    virtual std::size_t get_num_free_slots() const = 0;

    /// @return A free slot, or std::nullopt when the backend is full.
    virtual std::optional<std::size_t> acquire_slot() = 0;

    /// Returns a previously acquired slot to the free pool. Its contents are no longer valid to read.
    virtual void release_slot(std::size_t slot_id) = 0;

    /**
     * @brief Writes one slot's data.
     * @param hash The prefix-cache hash this slot's contents belong to, supplied purely so a backend that
     * supports cross-restart persistence can record it in its own index; the backend must never use it
     * for any eviction, lookup, or prefix-cache policy decision - `KVCacheOffloadCache` alone owns that.
     * Omit it to leave the slot outside any persisted index (still valid for the current process).
     */
    virtual void write_slot(std::size_t slot_id,
                            const std::vector<std::uint8_t>& data,
                            std::optional<std::size_t> hash = std::nullopt) = 0;

    /**
     * @brief Reads one slot's data.
     * @throws ov::Exception if the backend detects the data is not trustworthy (e.g. a checksum
     * mismatch), so a caller that already treats I/O errors as a cache miss gets the same safe behavior.
     */
    virtual void read_slot(std::size_t slot_id, std::vector<std::uint8_t>& data) const = 0;

    /// @return (hash, slot_id) pairs recovered from a compatible persisted cache at construction time.
    /// Always empty for a backend that does not support persistence.
    virtual const std::vector<std::pair<std::size_t, std::size_t>>& get_recovered_entries() const = 0;

    /// @return A short human-readable description (e.g. a file path) for logging/diagnostics only; never
    /// parsed or relied upon for behavior.
    virtual std::string describe() const = 0;
};

}  // namespace ov::genai
