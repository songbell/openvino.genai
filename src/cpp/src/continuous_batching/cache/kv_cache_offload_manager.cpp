// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/kv_cache_offload_manager.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#include <random>

#include "logger.hpp"

#ifdef _WIN32
#    include <fcntl.h>
#    include <io.h>
#    include <process.h>
#    include <share.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#else
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif

#include "openvino/core/except.hpp"

namespace ov::genai {

namespace {

constexpr size_t MAX_FILE_NAME_ATTEMPTS = 16;

// On-disk manifest layout. Integers are written in the host's native byte order: a persisted cache is
// only ever expected to be reopened on the machine (and build) that wrote it, so this keeps
// (de)serialization to plain memcpy instead of an explicit wire format.
constexpr uint32_t MANIFEST_MAGIC = 0x564B564Fu;  // "OVKV"
constexpr uint32_t MANIFEST_FORMAT_VERSION = 2;
constexpr size_t MANIFEST_HEADER_SIZE = 56;
constexpr size_t MANIFEST_RECORD_SIZE = 24;

constexpr const char* PERSISTENT_DATA_FILE_NAME = "ov_genai_kv_offload.data";
constexpr const char* PERSISTENT_MANIFEST_FILE_NAME = "ov_genai_kv_offload.manifest";

uint64_t fnv1a_64(const uint8_t* data, size_t size, uint64_t seed = 0xcbf29ce484222325ULL) {
    uint64_t hash = seed;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

uint64_t fnv1a_64(const std::string& text) {
    return fnv1a_64(reinterpret_cast<const uint8_t*>(text.data()), text.size());
}

/// Identifies the KV byte layout (layer count and each layer's key/value segment size) so a persisted
/// cache is rejected if the model's precision or shapes changed since it was written, even if the caller
/// supplied the same model_fingerprint string by mistake.
uint64_t compute_layout_fingerprint(const KVCacheDiskLayout& layout) {
    const size_t num_layers = layout.get_num_layers();
    uint64_t hash = fnv1a_64(reinterpret_cast<const uint8_t*>(&num_layers), sizeof(num_layers));
    for (size_t i = 0; i < num_layers; ++i) {
        const auto key_segment_size = layout.get_key_segment(i).size;
        const auto value_segment_size = layout.get_value_segment(i).size;
        hash = fnv1a_64(reinterpret_cast<const uint8_t*>(&key_segment_size), sizeof(key_segment_size), hash);
        hash = fnv1a_64(reinterpret_cast<const uint8_t*>(&value_segment_size), sizeof(value_segment_size), hash);
    }
    return hash;
}

std::string make_offload_file_name() {
    static std::atomic<uint64_t> counter{0};
#ifdef _WIN32
    const auto pid = static_cast<unsigned long long>(_getpid());
#else
    const auto pid = static_cast<unsigned long long>(::getpid());
#endif
    std::random_device random_device;
    return "ov_genai_kv_offload_" + std::to_string(pid) + "_" +
           std::to_string(counter.fetch_add(1)) + "_" + std::to_string(random_device()) + ".bin";
}

int create_exclusive_file(const std::filesystem::path& path) {
#ifdef _WIN32
    int fd = -1;
    const errno_t error = _wsopen_s(&fd,
                                    path.wstring().c_str(),
                                    _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY | _O_RANDOM,
                                    _SH_DENYRW,
                                    _S_IREAD | _S_IWRITE);
    return error == 0 ? fd : -1;
#else
    return ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
#endif
}

/// Opens an existing file for read/write, or returns -1 (without throwing) if it isn't there.
int open_existing_file(const std::filesystem::path& path) {
#ifdef _WIN32
    int fd = -1;
    const errno_t error = _wsopen_s(&fd, path.wstring().c_str(), _O_RDWR | _O_BINARY, _SH_DENYRW, 0);
    return error == 0 ? fd : -1;
#else
    return ::open(path.c_str(), O_RDWR);
#endif
}

/// Creates a file for read/write, discarding any previous contents at that path.
int create_or_truncate_file(const std::filesystem::path& path) {
#ifdef _WIN32
    int fd = -1;
    const errno_t error = _wsopen_s(&fd,
                                    path.wstring().c_str(),
                                    _O_RDWR | _O_CREAT | _O_TRUNC | _O_BINARY,
                                    _SH_DENYRW,
                                    _S_IREAD | _S_IWRITE);
    return error == 0 ? fd : -1;
#else
    return ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
#endif
}

void resize_file(int fd, size_t size) {
#ifdef _WIN32
    const int result = _chsize_s(fd, static_cast<__int64>(size));
    const bool failed = result != 0;
#else
    const bool failed = ::ftruncate(fd, static_cast<off_t>(size)) != 0;
#endif
    OPENVINO_ASSERT(!failed, "Failed to reserve ", size, " bytes for the KV cache offload file: ", std::strerror(errno));
}

void close_file(int fd) {
#ifdef _WIN32
    _close(fd);
#else
    ::close(fd);
#endif
}

/// Blocks until the file's contents are durable, so a persisted cache survives more than just this
/// process exiting (e.g. an OS crash or power loss), not only a plain crash where the page cache is
/// still intact for the next process to read.
void fsync_file(int fd) {
#ifdef _WIN32
    _commit(fd);
#else
    ::fsync(fd);
#endif
}

size_t file_size_or_zero(const std::filesystem::path& path) {
    std::error_code error_code;
    const auto size = std::filesystem::file_size(path, error_code);
    return error_code ? 0 : static_cast<size_t>(size);
}

}  // namespace

bool KVCacheOffloadManager::is_supported_device(const std::string& device) {
    // Device caches are staged through host tensors, so anything the KV cache manager can copy a block
    // out of works; NPU keeps its cache outside of this manager and is therefore excluded.
    return device.find("CPU") != std::string::npos || device.find("GPU") != std::string::npos;
}

void KVCacheOffloadManager::remove_persisted_cache(const std::filesystem::path& directory) {
    std::error_code error_code;
    std::filesystem::remove(directory / PERSISTENT_DATA_FILE_NAME, error_code);
    std::filesystem::remove(directory / PERSISTENT_MANIFEST_FILE_NAME, error_code);
}

KVCacheOffloadManager::KVCacheOffloadManager(const KVCacheDiskLayout& layout,
                                             const CacheOffloadConfig& config,
                                             uint64_t tenant_isolation_seed)
    : m_layout(layout), m_tenant_isolation_seed(tenant_isolation_seed) {
    m_slot_size = m_layout.get_slot_size();
    OPENVINO_ASSERT(m_slot_size > 0, "KV cache offload slot size must be greater than 0");
    m_layer_spans.reserve(m_layout.get_num_layers());
    for (size_t layer = 0; layer < m_layout.get_num_layers(); ++layer) {
        const auto key = m_layout.get_key_segment(layer);
        const auto value = m_layout.get_value_segment(layer);
        OPENVINO_ASSERT(key.offset + key.size == value.offset,
                        "KV cache disk layout expects each layer's value segment to immediately follow "
                        "its key segment");
        m_layer_spans.emplace_back(key.offset, key.size + value.size);
    }

    StorageModelMetadata metadata;
    metadata.model_fingerprint = config.model_fingerprint;
    metadata.tokenizer_hash = config.tokenizer_fingerprint;
    metadata.num_layers = m_layout.get_num_layers();

    const size_t effective_capacity_bytes =
        config.capacity_bytes > 0 ? config.capacity_bytes : config.storage_cache_size * 1024ULL * 1024ULL * 1024ULL;
    m_num_slots = effective_capacity_bytes / m_slot_size;
    OPENVINO_ASSERT(m_num_slots > 0,
                    "KV cache offload capacity of ",
                    effective_capacity_bytes,
                    " bytes is smaller than a single cache block of ",
                    m_slot_size,
                    " bytes");

    initialize(config.storage_cache_dir, metadata, config.storage_plugin_properties);

    m_persistent = config.enable_persistence;
    if (m_persistent) {
        OPENVINO_ASSERT(!config.storage_cache_dir.empty(),
                        "KV cache offload persistence requires an explicit directory "
                        "(CacheOffloadConfig::storage_cache_dir); the system temporary directory is not "
                        "appropriate for data meant to outlive the run");
        OPENVINO_ASSERT(!config.model_fingerprint.empty() && !config.tokenizer_fingerprint.empty(),
                        "KV cache offload persistence requires non-empty model_fingerprint and tokenizer_fingerprint");
    }
    OPENVINO_ASSERT(config.use_page_cache,
                    "KV cache disk offload with direct I/O is not implemented yet, set use_page_cache to true");
    // Choosing between this backend and a plugin backend is the caller's job (see
    // CacheOrchestrator::enable_kv_cache_offload); these are a defensive check against constructing this
    // backend directly with a config that says otherwise, not the actual dispatch point.
    OPENVINO_ASSERT(config.storage_backend_type == "default",
                    "KVCacheOffloadManager (the default file-based backend) must only be constructed for "
                    "storage_backend_type 'default', got '",
                    config.storage_backend_type,
                    "'");
    OPENVINO_ASSERT(config.storage_plugin_path.empty(),
                    "KVCacheOffloadManager (the default file-based backend) must not be constructed when "
                    "storage_plugin_path is set; use storage_backend_type 'plugin' instead");

    std::filesystem::path directory;
    if (config.storage_cache_dir.empty()) {
        directory = std::filesystem::temp_directory_path();
    } else {
        directory = std::filesystem::path(config.storage_cache_dir);
        OPENVINO_ASSERT(std::filesystem::is_directory(directory),
                        "KV cache offload path '",
                        config.storage_cache_dir,
                        "' is not an existing directory");
    }

    m_slot_checksums.assign(m_num_slots, 0);

    if (m_persistent) {
        m_file_path = directory / PERSISTENT_DATA_FILE_NAME;
        m_manifest_path = directory / PERSISTENT_MANIFEST_FILE_NAME;
        if (!try_recover_persisted_cache(config)) {
            create_fresh_persisted_files(config);
        }

        std::vector<bool> occupied(m_num_slots, false);
        for (const auto& entry : m_hash_to_slot) {
            occupied[entry.second] = true;
        }
        m_free_slots.reserve(m_num_slots);
        for (size_t slot_id = m_num_slots; slot_id > 0; --slot_id) {
            if (!occupied[slot_id - 1]) {
                m_free_slots.push_back(slot_id - 1);
            }
        }
        return;
    }

    for (size_t attempt = 0; attempt < MAX_FILE_NAME_ATTEMPTS && m_fd < 0; ++attempt) {
        const std::filesystem::path candidate = directory / make_offload_file_name();
        const int fd = create_exclusive_file(candidate);
        if (fd >= 0) {
            m_fd = fd;
            m_file_path = candidate;
        } else if (errno != EEXIST) {
            OPENVINO_THROW("Failed to create the KV cache offload file in '",
                           directory.string(),
                           "': ",
                           std::strerror(errno));
        }
    }
    OPENVINO_ASSERT(m_fd >= 0,
                    "Failed to create a unique KV cache offload file in '",
                    directory.string(),
                    "'");

    try {
        resize_file(m_fd, m_num_slots * m_slot_size);
    } catch (...) {
        close_and_remove();
        throw;
    }

    m_free_slots.reserve(m_num_slots);
    for (size_t slot_id = m_num_slots; slot_id > 0; --slot_id) {
        m_free_slots.push_back(slot_id - 1);
    }
}

void KVCacheOffloadManager::initialize(const std::string& storage_path,
                                       const StorageModelMetadata& metadata,
                                       const ov::AnyMap& custom_properties) {
    // Construction already performs the real setup (see the constructor above): the byte layout comes
    // from the KVCacheDiskLayout given at construction time, not from `metadata`'s (necessarily coarser,
    // uniform-layer-size) fields, which exist for interface parity with a future third-party plugin that
    // has no access to that C++ type. Only a consistency sanity check is done here.
    OPENVINO_ASSERT(metadata.num_layers == 0 || metadata.num_layers == m_layout.get_num_layers(),
                    "KV cache offload metadata.num_layers (",
                    metadata.num_layers,
                    ") does not match the configured layout (",
                    m_layout.get_num_layers(),
                    ")");
    (void)storage_path;
    (void)custom_properties;
}

bool KVCacheOffloadManager::try_recover_persisted_cache(const CacheOffloadConfig& config) {
    if (!std::filesystem::exists(m_file_path) || !std::filesystem::exists(m_manifest_path)) {
        // Either nothing has been persisted here yet, or a previous run was interrupted before both
        // files existed; either way there is nothing safe to recover.
        return false;
    }

    const size_t expected_manifest_size = MANIFEST_HEADER_SIZE + m_num_slots * MANIFEST_RECORD_SIZE;
    if (file_size_or_zero(m_manifest_path) != expected_manifest_size ||
        file_size_or_zero(m_file_path) != m_num_slots * m_slot_size) {
        GENAI_WARN("KV cache offload persisted cache in '%s' has an unexpected file size, rebuilding it fresh",
                   m_file_path.parent_path().string().c_str());
        return false;
    }

    const int manifest_fd = open_existing_file(m_manifest_path);
    if (manifest_fd < 0) {
        return false;
    }

    std::vector<uint8_t> header(MANIFEST_HEADER_SIZE);
    read_at(manifest_fd, 0, header.data(), header.size());

    uint32_t magic = 0, format_version = 0;
    uint64_t header_slot_size = 0, header_num_slots = 0, model_fp = 0, tokenizer_fp = 0, layout_fp = 0, tenant_fp = 0;
    std::memcpy(&magic, header.data() + 0, sizeof(magic));
    std::memcpy(&format_version, header.data() + 4, sizeof(format_version));
    std::memcpy(&header_slot_size, header.data() + 8, sizeof(header_slot_size));
    std::memcpy(&header_num_slots, header.data() + 16, sizeof(header_num_slots));
    std::memcpy(&model_fp, header.data() + 24, sizeof(model_fp));
    std::memcpy(&tokenizer_fp, header.data() + 32, sizeof(tokenizer_fp));
    std::memcpy(&layout_fp, header.data() + 40, sizeof(layout_fp));
    std::memcpy(&tenant_fp, header.data() + 48, sizeof(tenant_fp));

    const bool compatible = magic == MANIFEST_MAGIC && format_version == MANIFEST_FORMAT_VERSION &&
                            header_slot_size == m_slot_size && header_num_slots == m_num_slots &&
                            model_fp == fnv1a_64(config.model_fingerprint) &&
                            tokenizer_fp == fnv1a_64(config.tokenizer_fingerprint) &&
                            layout_fp == compute_layout_fingerprint(m_layout) &&
                            tenant_fp == m_tenant_isolation_seed;
    if (!compatible) {
        close_file(manifest_fd);
        GENAI_WARN("KV cache offload persisted cache in '%s' is incompatible with the current model/tokenizer/layout, "
                   "rebuilding it fresh",
                   m_file_path.parent_path().string().c_str());
        return false;
    }

    const int data_fd = open_existing_file(m_file_path);
    if (data_fd < 0) {
        close_file(manifest_fd);
        return false;
    }

    m_manifest_fd = manifest_fd;
    m_fd = data_fd;

    std::vector<uint8_t> record(MANIFEST_RECORD_SIZE);
    for (size_t slot_id = 0; slot_id < m_num_slots; ++slot_id) {
        read_at(m_manifest_fd, MANIFEST_HEADER_SIZE + slot_id * MANIFEST_RECORD_SIZE, record.data(), record.size());
        uint64_t recorded_hash = 0, recorded_checksum = 0;
        uint32_t valid = 0;
        std::memcpy(&recorded_hash, record.data() + 0, sizeof(recorded_hash));
        std::memcpy(&recorded_checksum, record.data() + 8, sizeof(recorded_checksum));
        std::memcpy(&valid, record.data() + 16, sizeof(valid));
        if (valid == 1) {
            m_slot_checksums[slot_id] = recorded_checksum;
            m_hash_to_slot[recorded_hash] = slot_id;
            m_recovered_hashes.push_back(recorded_hash);
        }
    }

    GENAI_INFO("[KV_TRACE] KVCacheOffloadManager recover_persisted_cache slots=%zu recovered=%zu",
               m_num_slots,
               m_recovered_hashes.size());
    return true;
}

void KVCacheOffloadManager::create_fresh_persisted_files(const CacheOffloadConfig& config) {
    if (m_manifest_fd >= 0) {
        close_file(m_manifest_fd);
        m_manifest_fd = -1;
    }
    if (m_fd >= 0) {
        close_file(m_fd);
        m_fd = -1;
    }
    m_recovered_hashes.clear();
    m_hash_to_slot.clear();
    std::fill(m_slot_checksums.begin(), m_slot_checksums.end(), 0);

    m_fd = create_or_truncate_file(m_file_path);
    OPENVINO_ASSERT(m_fd >= 0,
                    "Failed to create the KV cache offload data file '",
                    m_file_path.string(),
                    "': ",
                    std::strerror(errno));
    try {
        resize_file(m_fd, m_num_slots * m_slot_size);
    } catch (...) {
        close_file(m_fd);
        m_fd = -1;
        throw;
    }

    m_manifest_fd = create_or_truncate_file(m_manifest_path);
    if (m_manifest_fd < 0) {
        close_file(m_fd);
        m_fd = -1;
        OPENVINO_THROW("Failed to create the KV cache offload manifest file '",
                       m_manifest_path.string(),
                       "': ",
                       std::strerror(errno));
    }

    std::vector<uint8_t> header(MANIFEST_HEADER_SIZE, 0);
    const uint32_t magic = MANIFEST_MAGIC;
    const uint32_t format_version = MANIFEST_FORMAT_VERSION;
    const uint64_t header_slot_size = m_slot_size;
    const uint64_t header_num_slots = m_num_slots;
    const uint64_t model_fp = fnv1a_64(config.model_fingerprint);
    const uint64_t tokenizer_fp = fnv1a_64(config.tokenizer_fingerprint);
    const uint64_t layout_fp = compute_layout_fingerprint(m_layout);
    const uint64_t tenant_fp = m_tenant_isolation_seed;
    std::memcpy(header.data() + 0, &magic, sizeof(magic));
    std::memcpy(header.data() + 4, &format_version, sizeof(format_version));
    std::memcpy(header.data() + 8, &header_slot_size, sizeof(header_slot_size));
    std::memcpy(header.data() + 16, &header_num_slots, sizeof(header_num_slots));
    std::memcpy(header.data() + 24, &model_fp, sizeof(model_fp));
    std::memcpy(header.data() + 32, &tokenizer_fp, sizeof(tokenizer_fp));
    std::memcpy(header.data() + 40, &layout_fp, sizeof(layout_fp));
    std::memcpy(header.data() + 48, &tenant_fp, sizeof(tenant_fp));
    write_at(m_manifest_fd, 0, header.data(), header.size());

    const std::vector<uint8_t> empty_record(MANIFEST_RECORD_SIZE, 0);  // hash=0, checksum=0, valid=0
    for (size_t slot_id = 0; slot_id < m_num_slots; ++slot_id) {
        write_at(m_manifest_fd, MANIFEST_HEADER_SIZE + slot_id * MANIFEST_RECORD_SIZE, empty_record.data(), empty_record.size());
    }
    fsync_file(m_manifest_fd);

    GENAI_INFO("[KV_TRACE] KVCacheOffloadManager create_persisted_cache slots=%zu", m_num_slots);
}

KVCacheOffloadManager::~KVCacheOffloadManager() {
    shutdown();
}

void KVCacheOffloadManager::shutdown() {
    if (m_persistent) {
        // Files intentionally survive the process so a later run can recover them; only release the fds.
        if (m_manifest_fd >= 0) {
            close_file(m_manifest_fd);
            m_manifest_fd = -1;
        }
        if (m_fd >= 0) {
            close_file(m_fd);
            m_fd = -1;
        }
        return;
    }
    close_and_remove();
}

void KVCacheOffloadManager::close_and_remove() noexcept {
    if (m_fd >= 0) {
        close_file(m_fd);
        m_fd = -1;
    }
    if (!m_file_path.empty()) {
        std::error_code error_code;
        std::filesystem::remove(m_file_path, error_code);
        m_file_path.clear();
    }
}

size_t KVCacheOffloadManager::get_num_free_slots() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_free_slots.size();
}

bool KVCacheOffloadManager::has_block(uint64_t block_hash) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_hash_to_slot.find(block_hash) != m_hash_to_slot.end();
}

std::vector<bool> KVCacheOffloadManager::has_blocks(const std::vector<uint64_t>& block_hashes) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<bool> result(block_hashes.size(), false);
    for (size_t i = 0; i < block_hashes.size(); ++i) {
        result[i] = m_hash_to_slot.find(block_hashes[i]) != m_hash_to_slot.end();
    }
    return result;
}

namespace {

/// Groups a batch of per-layer requests by block_hash, validating that each group covers every layer
/// exactly once (the whole-block-per-call contract `IKVCacheStorageBackend` documents).
std::unordered_map<uint64_t, std::vector<const BlockIORequest*>> group_by_hash(
    const std::vector<BlockIORequest>& requests,
    size_t num_layers) {
    std::unordered_map<uint64_t, std::vector<const BlockIORequest*>> groups;
    for (const auto& request : requests) {
        groups[request.block_hash].push_back(&request);
    }
    for (const auto& [hash, group] : groups) {
        OPENVINO_ASSERT(group.size() == num_layers,
                        "KV cache offload expects exactly ",
                        num_layers,
                        " layer requests for hash ",
                        hash,
                        " in one call, got ",
                        group.size());
        std::vector<bool> seen_layers(num_layers, false);
        for (const auto* request : group) {
            OPENVINO_ASSERT(request->layer_idx < num_layers, "Invalid KV cache offload layer_idx ", request->layer_idx);
            OPENVINO_ASSERT(!seen_layers[request->layer_idx],
                            "KV cache offload got duplicate layer_idx ",
                            request->layer_idx,
                            " for hash ",
                            hash);
            seen_layers[request->layer_idx] = true;
        }
    }
    return groups;
}

}  // namespace

bool KVCacheOffloadManager::write_blocks(const std::vector<BlockIORequest>& requests) {
    if (requests.empty()) {
        return true;
    }
    const auto groups = group_by_hash(requests, m_layer_spans.size());

    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& [hash, group] : groups) {
        size_t slot_id;
        auto existing = m_hash_to_slot.find(hash);
        if (existing != m_hash_to_slot.end()) {
            slot_id = existing->second;
        } else {
            OPENVINO_ASSERT(!m_free_slots.empty(),
                            "KV cache offload write_blocks called for a new hash with no free slots; "
                            "the caller must evict before writing when at capacity");
            slot_id = m_free_slots.back();
            m_free_slots.pop_back();
            m_hash_to_slot[hash] = slot_id;
        }

        const size_t slot_offset = m_layout.get_slot_offset(slot_id);
        uint64_t checksum = 0xcbf29ce484222325ULL;
        // Sorted by layer_idx so the whole-block checksum is independent of the caller's request order.
        std::vector<const BlockIORequest*> ordered(group);
        std::sort(ordered.begin(), ordered.end(), [](const BlockIORequest* a, const BlockIORequest* b) {
            return a->layer_idx < b->layer_idx;
        });
        for (const auto* request : ordered) {
            const auto& span = m_layer_spans[request->layer_idx];
            OPENVINO_ASSERT(request->buffer_size_bytes == span.second,
                            "Unexpected KV cache offload layer size for layer ",
                            request->layer_idx,
                            ": got ",
                            request->buffer_size_bytes,
                            ", expected ",
                            span.second);
            const auto* data = static_cast<const uint8_t*>(request->host_buffer);
            write_at(m_fd, slot_offset + span.first, data, request->buffer_size_bytes);
            checksum = fnv1a_64(data, request->buffer_size_bytes, checksum);
        }
        m_slot_checksums[slot_id] = checksum;
        GENAI_INFO("[KV_TRACE] KVCacheOffloadManager write_blocks hash=%llu slot=%zu layers=%zu",
                   static_cast<unsigned long long>(hash),
                   slot_id,
                   ordered.size());

        if (m_persistent) {
            fsync_file(m_fd);
            write_manifest_record(slot_id, hash, checksum);
        }
    }
    return true;
}

bool KVCacheOffloadManager::read_blocks(const std::vector<BlockIORequest>& requests) {
    if (requests.empty()) {
        return true;
    }
    const auto groups = group_by_hash(requests, m_layer_spans.size());

    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& [hash, group] : groups) {
        auto it = m_hash_to_slot.find(hash);
        if (it == m_hash_to_slot.end()) {
            return false;
        }
        const size_t slot_id = it->second;
        const size_t slot_offset = m_layout.get_slot_offset(slot_id);

        std::vector<const BlockIORequest*> ordered(group);
        std::sort(ordered.begin(), ordered.end(), [](const BlockIORequest* a, const BlockIORequest* b) {
            return a->layer_idx < b->layer_idx;
        });
        uint64_t checksum = 0xcbf29ce484222325ULL;
        for (const auto* request : ordered) {
            const auto& span = m_layer_spans[request->layer_idx];
            OPENVINO_ASSERT(request->buffer_size_bytes == span.second,
                            "Unexpected KV cache offload layer size for layer ",
                            request->layer_idx,
                            ": got ",
                            request->buffer_size_bytes,
                            ", expected ",
                            span.second);
            auto* data = static_cast<uint8_t*>(request->host_buffer);
            read_at(m_fd, slot_offset + span.first, data, request->buffer_size_bytes);
            checksum = fnv1a_64(data, request->buffer_size_bytes, checksum);
        }
        if (checksum != m_slot_checksums[slot_id]) {
            GENAI_WARN("KV cache offload slot %zu (hash %llu) failed its integrity check; contents may be corrupted",
                       slot_id,
                       static_cast<unsigned long long>(hash));
            return false;
        }
        GENAI_INFO("[KV_TRACE] KVCacheOffloadManager read_blocks hash=%llu slot=%zu layers=%zu",
                   static_cast<unsigned long long>(hash),
                   slot_id,
                   ordered.size());
    }
    return true;
}

void KVCacheOffloadManager::evict_blocks(const std::vector<uint64_t>& block_hashes) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto hash : block_hashes) {
        auto it = m_hash_to_slot.find(hash);
        if (it == m_hash_to_slot.end()) {
            continue;
        }
        const size_t slot_id = it->second;
        m_hash_to_slot.erase(it);
        m_free_slots.push_back(slot_id);
        if (m_persistent) {
            mark_manifest_invalid(slot_id);
        }
        GENAI_INFO("[KV_TRACE] KVCacheOffloadManager evict_blocks hash=%llu slot=%zu free_after=%zu",
                   static_cast<unsigned long long>(hash),
                   slot_id,
                   m_free_slots.size());
    }
}

void KVCacheOffloadManager::flush_manifest() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_persistent && m_manifest_fd >= 0) {
        fsync_file(m_manifest_fd);
    }
}

void KVCacheOffloadManager::mark_manifest_invalid(size_t slot_id) {
    const size_t offset = MANIFEST_HEADER_SIZE + slot_id * MANIFEST_RECORD_SIZE + 16;
    const uint32_t valid = 0;
    write_at(m_manifest_fd, offset, reinterpret_cast<const uint8_t*>(&valid), sizeof(valid));
    fsync_file(m_manifest_fd);
}

void KVCacheOffloadManager::write_manifest_record(size_t slot_id, uint64_t hash, uint64_t checksum) {
    const size_t offset = MANIFEST_HEADER_SIZE + slot_id * MANIFEST_RECORD_SIZE;
    uint8_t body[16];
    std::memcpy(body + 0, &hash, sizeof(hash));
    std::memcpy(body + 8, &checksum, sizeof(checksum));
    write_at(m_manifest_fd, offset, body, sizeof(body));
    // Durable before the valid flag flips, so a crash in between leaves the slot reading back as absent
    // (valid still 0) rather than valid with a hash/checksum pair that may not have made it to disk.
    fsync_file(m_manifest_fd);

    const uint32_t valid = 1;
    write_at(m_manifest_fd, offset + 16, reinterpret_cast<const uint8_t*>(&valid), sizeof(valid));
    fsync_file(m_manifest_fd);
}

void KVCacheOffloadManager::write_at(int fd, size_t offset, const uint8_t* data, size_t size) const {
    size_t written = 0;
    while (written < size) {
        const size_t remaining = size - written;
#ifdef _WIN32
        OPENVINO_ASSERT(_lseeki64(fd, static_cast<__int64>(offset + written), SEEK_SET) >= 0,
                        "Failed to seek in the KV cache offload file: ",
                        std::strerror(errno));
        const auto chunk =
            static_cast<unsigned int>(std::min<size_t>(remaining, std::numeric_limits<int>::max()));
        const int result = _write(fd, data + written, chunk);
#else
        const ssize_t result =
            ::pwrite(fd, data + written, remaining, static_cast<off_t>(offset + written));
#endif
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            OPENVINO_THROW("Failed to write the KV cache offload file: ", std::strerror(errno));
        }
        OPENVINO_ASSERT(result > 0,
                        "Short write to the KV cache offload file: wrote ",
                        written,
                        " of ",
                        size,
                        " bytes");
        written += static_cast<size_t>(result);
    }
}

void KVCacheOffloadManager::read_at(int fd, size_t offset, uint8_t* data, size_t size) const {
    size_t read_bytes = 0;
    while (read_bytes < size) {
        const size_t remaining = size - read_bytes;
#ifdef _WIN32
        OPENVINO_ASSERT(_lseeki64(fd, static_cast<__int64>(offset + read_bytes), SEEK_SET) >= 0,
                        "Failed to seek in the KV cache offload file: ",
                        std::strerror(errno));
        const auto chunk =
            static_cast<unsigned int>(std::min<size_t>(remaining, std::numeric_limits<int>::max()));
        const int result = _read(fd, data + read_bytes, chunk);
#else
        const ssize_t result =
            ::pread(fd, data + read_bytes, remaining, static_cast<off_t>(offset + read_bytes));
#endif
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            OPENVINO_THROW("Failed to read the KV cache offload file: ", std::strerror(errno));
        }
        OPENVINO_ASSERT(result > 0,
                        "Short read from the KV cache offload file: read ",
                        read_bytes,
                        " of ",
                        size,
                        " bytes");
        read_bytes += static_cast<size_t>(result);
    }
}

}  // namespace ov::genai

