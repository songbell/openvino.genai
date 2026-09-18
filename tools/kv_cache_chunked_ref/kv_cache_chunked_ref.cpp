// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Reference greedy_causal_lm-style pipeline for exercising SchedulerConfig::use_chunked_kv_cache.
// Not a product sample: a standalone tool to reproduce the chunked KV cache path end-to-end and
// collect TTFT/TPOT/throughput numbers for the kernel optimization work described in
// chunked_kv_cache_kernel_optimization_handoff.md.

#include <cstdlib>
#include <iostream>
#include <string>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"

namespace {

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " <MODEL_DIR> \"<PROMPT>\" [DEVICE] [CHUNKED=0|1] [CHUNK_SIZE_BLOCKS] [MAX_NEW_TOKENS]\n"
              << "  DEVICE             default: GPU\n"
              << "  CHUNKED            default: 1 (use_chunked_kv_cache)\n"
              << "  CHUNK_SIZE_BLOCKS  default: 16 (kv_cache_chunk_size_blocks)\n"
              << "  MAX_NEW_TOKENS     default: 128\n";
}

ov::genai::SchedulerConfig make_scheduler_config(bool use_chunked_kv_cache, size_t chunk_size_blocks) {
    ov::genai::SchedulerConfig config;
    config.num_kv_blocks = 0;  // let the scheduler grow the cache dynamically
    config.max_num_batched_tokens = 512;
    config.max_num_seqs = 2;
    config.dynamic_split_fuse = false;
    config.use_chunked_kv_cache = use_chunked_kv_cache;
    config.kv_cache_chunk_size_blocks = chunk_size_blocks;
    return config;
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string models_path = argv[1];
    const std::string prompt = argv[2];
    const std::string device = argc > 3 ? argv[3] : "GPU";
    const bool use_chunked_kv_cache = argc > 4 ? std::stoi(argv[4]) != 0 : true;
    const size_t chunk_size_blocks = argc > 5 ? static_cast<size_t>(std::stoul(argv[5])) : 16;
    const size_t max_new_tokens = argc > 6 ? static_cast<size_t>(std::stoul(argv[6])) : 128;

    std::cout << "Model: " << models_path << "\n"
              << "Device: " << device << "\n"
              << "use_chunked_kv_cache: " << std::boolalpha << use_chunked_kv_cache << "\n"
              << "kv_cache_chunk_size_blocks: " << chunk_size_blocks << "\n"
              << "max_new_tokens: " << max_new_tokens << "\n";

    // Chunked KV cache currently only supports the uncompressed cache path, so KV_CACHE_PRECISION
    // must be forced to f16 (GPU otherwise defaults to a quantized cache).
    ov::AnyMap properties{{"KV_CACHE_PRECISION", ov::element::f16}};
    ov::genai::ContinuousBatchingPipeline pipeline(models_path,
                                                    make_scheduler_config(use_chunked_kv_cache, chunk_size_blocks),
                                                    device,
                                                    properties);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = max_new_tokens;
    generation_config.do_sample = false;
    generation_config.num_beams = 1;

    auto results = pipeline.generate({prompt}, {generation_config});
    if (results.empty() || results.front().m_generation_ids.empty()) {
        std::cerr << "Pipeline produced no text\n";
        return EXIT_FAILURE;
    }

    const auto& result = results.front();
    std::cout << "\n--- Generated text ---\n" << result.m_generation_ids.front() << "\n";

    auto perf = result.perf_metrics;
    std::cout << "\n--- Perf metrics ---\n"
              << "TTFT, ms: " << perf.get_ttft().mean << "\n"
              << "TPOT, ms/token: " << perf.get_tpot().mean << "\n"
              << "Throughput, tokens/s: " << perf.get_throughput().mean << "\n"
              << "Generate duration, ms: " << perf.get_generate_duration().mean << "\n";

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
