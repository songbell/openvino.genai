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
constexpr uint32_t MANIFEST_FORMAT_VERSION = 1;
constexpr size_t MANIFEST_HEADER_SIZE = 48;
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
                                             const std::string& device)
    : m_layout(layout), m_persistent(config.enable_persistence) {
    OPENVINO_ASSERT(is_supported_device(device),
                    "KV cache disk offload is implemented for CPU and GPU, but the inference device is '",
                    device,
                    "'");
    OPENVINO_ASSERT(config.use_page_cache,
                    "KV cache disk offload with direct I/O is not implemented yet, set use_page_cache to true");

    m_slot_size = m_layout.get_slot_size();
    OPENVINO_ASSERT(m_slot_size > 0, "KV cache offload slot size must be greater than 0");

    m_num_slots = config.capacity_bytes / m_slot_size;
    OPENVINO_ASSERT(m_num_slots > 0,
                    "KV cache offload capacity of ",
                    config.capacity_bytes,
                    " bytes is smaller than a single cache block of ",
                    m_slot_size,
                    " bytes");

    if (m_persistent) {
        OPENVINO_ASSERT(!config.path.empty(),
                        "KV cache offload persistence requires an explicit directory (CacheOffloadConfig::path); "
                        "the system temporary directory is not appropriate for data meant to outlive the run");
        OPENVINO_ASSERT(!config.model_fingerprint.empty() && !config.tokenizer_fingerprint.empty(),
                        "KV cache offload persistence requires non-empty model_fingerprint and tokenizer_fingerprint");
    }

    std::filesystem::path directory;
    if (config.path.empty()) {
        directory = std::filesystem::temp_directory_path();
    } else {
        directory = std::filesystem::path(config.path);
        OPENVINO_ASSERT(std::filesystem::is_directory(directory),
                        "KV cache offload path '",
                        config.path,
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
        for (const auto& entry : m_recovered_entries) {
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
    uint64_t header_slot_size = 0, header_num_slots = 0, model_fp = 0, tokenizer_fp = 0, layout_fp = 0;
    std::memcpy(&magic, header.data() + 0, sizeof(magic));
    std::memcpy(&format_version, header.data() + 4, sizeof(format_version));
    std::memcpy(&header_slot_size, header.data() + 8, sizeof(header_slot_size));
    std::memcpy(&header_num_slots, header.data() + 16, sizeof(header_num_slots));
    std::memcpy(&model_fp, header.data() + 24, sizeof(model_fp));
    std::memcpy(&tokenizer_fp, header.data() + 32, sizeof(tokenizer_fp));
    std::memcpy(&layout_fp, header.data() + 40, sizeof(layout_fp));

    const bool compatible = magic == MANIFEST_MAGIC && format_version == MANIFEST_FORMAT_VERSION &&
                            header_slot_size == m_slot_size && header_num_slots == m_num_slots &&
                            model_fp == fnv1a_64(config.model_fingerprint) &&
                            tokenizer_fp == fnv1a_64(config.tokenizer_fingerprint) &&
                            layout_fp == compute_layout_fingerprint(m_layout);
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
            m_recovered_entries.emplace_back(static_cast<size_t>(recorded_hash), slot_id);
        }
    }

    GENAI_INFO("[KV_TRACE] KVCacheOffloadManager recover_persisted_cache slots=%zu recovered=%zu",
               m_num_slots,
               m_recovered_entries.size());
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
    m_recovered_entries.clear();
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
    std::memcpy(header.data() + 0, &magic, sizeof(magic));
    std::memcpy(header.data() + 4, &format_version, sizeof(format_version));
    std::memcpy(header.data() + 8, &header_slot_size, sizeof(header_slot_size));
    std::memcpy(header.data() + 16, &header_num_slots, sizeof(header_num_slots));
    std::memcpy(header.data() + 24, &model_fp, sizeof(model_fp));
    std::memcpy(header.data() + 32, &tokenizer_fp, sizeof(tokenizer_fp));
    std::memcpy(header.data() + 40, &layout_fp, sizeof(layout_fp));
    write_at(m_manifest_fd, 0, header.data(), header.size());

    const std::vector<uint8_t> empty_record(MANIFEST_RECORD_SIZE, 0);  // hash=0, checksum=0, valid=0
    for (size_t slot_id = 0; slot_id < m_num_slots; ++slot_id) {
        write_at(m_manifest_fd, MANIFEST_HEADER_SIZE + slot_id * MANIFEST_RECORD_SIZE, empty_record.data(), empty_record.size());
    }
    fsync_file(m_manifest_fd);

    GENAI_INFO("[KV_TRACE] KVCacheOffloadManager create_persisted_cache slots=%zu", m_num_slots);
}

KVCacheOffloadManager::~KVCacheOffloadManager() {
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

std::optional<size_t> KVCacheOffloadManager::acquire_slot() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_free_slots.empty()) {
        return std::nullopt;
    }
    const size_t slot_id = m_free_slots.back();
    m_free_slots.pop_back();
    GENAI_INFO("[KV_TRACE] KVCacheOffloadManager acquire_slot slot=%zu free_after=%zu",
               slot_id,
               m_free_slots.size());
    return slot_id;
}

void KVCacheOffloadManager::release_slot(size_t slot_id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    OPENVINO_ASSERT(slot_id < m_num_slots, "Invalid KV cache offload slot ", slot_id);
    OPENVINO_ASSERT(std::find(m_free_slots.begin(), m_free_slots.end(), slot_id) == m_free_slots.end(),
                    "KV cache offload slot ",
                    slot_id,
                    " is released twice");
    m_free_slots.push_back(slot_id);
    GENAI_INFO("[KV_TRACE] KVCacheOffloadManager release_slot slot=%zu free_after=%zu",
               slot_id,
               m_free_slots.size());
}

void KVCacheOffloadManager::write_slot(size_t slot_id, const std::vector<uint8_t>& block_data, std::optional<size_t> hash) {
    OPENVINO_ASSERT(slot_id < m_num_slots, "Invalid KV cache offload slot ", slot_id);
    OPENVINO_ASSERT(block_data.size() == m_slot_size,
                    "Unexpected KV cache offload block size: got ",
                    block_data.size(),
                    ", expected ",
                    m_slot_size);

    std::lock_guard<std::mutex> lock(m_mutex);
    GENAI_INFO("[KV_TRACE] KVCacheOffloadManager write_slot slot=%zu offset=%zu bytes=%zu",
               slot_id,
               m_layout.get_slot_offset(slot_id),
               block_data.size());
    write_at(m_fd, m_layout.get_slot_offset(slot_id), block_data.data(), m_slot_size);
    const uint64_t checksum = fnv1a_64(block_data.data(), block_data.size());
    m_slot_checksums[slot_id] = checksum;

    if (m_persistent && hash.has_value()) {
        fsync_file(m_fd);
        write_manifest_record(slot_id, *hash, checksum);
    }
}

void KVCacheOffloadManager::write_manifest_record(size_t slot_id, size_t hash, uint64_t checksum) {
    const size_t offset = MANIFEST_HEADER_SIZE + slot_id * MANIFEST_RECORD_SIZE;
    uint8_t body[16];
    const uint64_t hash64 = static_cast<uint64_t>(hash);
    std::memcpy(body + 0, &hash64, sizeof(hash64));
    std::memcpy(body + 8, &checksum, sizeof(checksum));
    write_at(m_manifest_fd, offset, body, sizeof(body));
    // Durable before the valid flag flips, so a crash in between leaves the slot reading back as absent
    // (valid still 0) rather than valid with a hash/checksum pair that may not have made it to disk.
    fsync_file(m_manifest_fd);

    const uint32_t valid = 1;
    write_at(m_manifest_fd, offset + 16, reinterpret_cast<const uint8_t*>(&valid), sizeof(valid));
    fsync_file(m_manifest_fd);
}

void KVCacheOffloadManager::read_slot(size_t slot_id, std::vector<uint8_t>& block_data) const {
    OPENVINO_ASSERT(slot_id < m_num_slots, "Invalid KV cache offload slot ", slot_id);
    block_data.resize(m_slot_size);

    std::lock_guard<std::mutex> lock(m_mutex);
    GENAI_INFO("[KV_TRACE] KVCacheOffloadManager read_slot slot=%zu offset=%zu bytes=%zu",
               slot_id,
               m_layout.get_slot_offset(slot_id),
               m_slot_size);
    read_at(m_fd, m_layout.get_slot_offset(slot_id), block_data.data(), m_slot_size);
    const uint64_t actual_checksum = fnv1a_64(block_data.data(), block_data.size());
    OPENVINO_ASSERT(actual_checksum == m_slot_checksums[slot_id],
                    "KV cache offload slot ",
                    slot_id,
                    " failed its integrity check; contents may be corrupted");
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

