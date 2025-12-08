// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include "read_prompt_from_file.h"
template <typename T>
void print_perf_metrics(T& perf_metrics, std::string model_name) {
    std::cout << "\n" << model_name << std::endl;
    auto generation_duration = perf_metrics.get_generate_duration().mean;
    std::cout << "  Generate time: " << generation_duration << " ms" << std::endl;
    std::cout << "  TTFT: " << perf_metrics.get_ttft().mean << " ± " << perf_metrics.get_ttft().std << " ms"
              << std::endl;
    std::cout << "  TPOT: " << perf_metrics.get_tpot().mean << " ± " << perf_metrics.get_tpot().std << " ms/token"
              << std::endl;
    std::cout << "  Num generated token: " << perf_metrics.get_num_generated_tokens() << " tokens" << std::endl;
    if (model_name == "Total") {
        std::cout << "  Total iteration number: " << perf_metrics.raw_metrics.m_new_token_times.size() << std::endl;
    } else {
        std::cout << "  Total iteration number: " << perf_metrics.raw_metrics.m_durations.size() << std::endl;
    }
    if (perf_metrics.get_num_input_tokens() > 0) {
        std::cout << "  Input token size: " << perf_metrics.get_num_input_tokens() << std::endl;
    }
}
int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\"");

    std::string models_path = argv[1];
    std::string prompt = argv[2];
    std::string device = "GPU";  // GPU can be used as well
    if (std::filesystem::is_regular_file(prompt)) {
        std::string prompt_file = prompt;
        prompt = utils::read_prompt(prompt_file);
    }

    ov::genai::LLMPipeline pipe(models_path, device, {ov::hint::kv_cache_precision(ov::element::f16)});
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 129;
    auto result = pipe.generate(prompt, config);
    print_perf_metrics(result.perf_metrics, "Total");
    std::cout << result << std::endl;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
