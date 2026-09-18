// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

namespace ov::genai {

namespace detail {

inline uint64_t fnv1a_64_for_isolation_seed(const uint8_t* data, size_t size) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

}  // namespace detail

/**
 * @brief Computes a stable 64-bit seed from a tenant identity, for mixing into prefix-cache block hashes
 * and (when persistence is enabled) the offload disk manifest, so two different tenants sharing the same
 * process, model and token content never compute the same hash and can never observe each other's cached
 * KV blocks, in memory or on disk.
 *
 * @return 0 exactly when both @p tenant_id and @p cache_salt are empty - the explicit "no tenant info
 * provided" case, which is only safe for single-tenant deployments: two callers that both leave tenant
 * identity unset are treated as the same (default) tenant, matching pre-isolation behavior. Any non-empty
 * @p tenant_id or @p cache_salt is guaranteed to produce a non-zero seed, so 0 unambiguously means
 * "unseeded" and can be used as a sentinel by callers.
 */
inline uint64_t compute_prefix_isolation_seed(const std::string& tenant_id, const std::string& cache_salt) {
    if (tenant_id.empty() && cache_salt.empty()) {
        return 0;
    }
    // A separator byte that cannot appear implicitly from concatenation ambiguity, e.g. tenant_id="a",
    // cache_salt="bc" must not hash the same as tenant_id="ab", cache_salt="c".
    std::string combined;
    combined.reserve(tenant_id.size() + cache_salt.size() + 1);
    combined.append(tenant_id);
    combined.push_back('\x1f');
    combined.append(cache_salt);

    const uint64_t seed =
        detail::fnv1a_64_for_isolation_seed(reinterpret_cast<const uint8_t*>(combined.data()), combined.size());
    // Guarantee non-zero so 0 stays an unambiguous "no tenant info" sentinel even in the astronomically
    // unlikely case that the hash of a non-empty identity happens to be exactly 0.
    return seed == 0 ? 1 : seed;
}

}  // namespace ov::genai
