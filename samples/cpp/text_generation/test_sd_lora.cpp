#include <openvino/openvino.hpp>
#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/speculative_decoding/perf_metrics.hpp>
#include <iostream>
#include <string>
#include <chrono>

static double throughput_excluding_first_token(ov::genai::PerfMetrics& metrics) {
    const auto gen = metrics.get_generate_duration();
    const auto ttft = metrics.get_ttft();
    const size_t num_tokens = metrics.get_num_generated_tokens();
    if (num_tokens <= 1) {
        return 0.0;
    }
    const double tail_ms = std::max(1.0, static_cast<double>(gen.mean - ttft.mean));
    return (num_tokens - 1) * 1000.0 / tail_ms;
}

int main() {
    try {
        std::string model_path = "/home/openvino-ci-97/bell/speculative_decoding/Qwen3-4B/";
        std::string draft_model_path = "/home/openvino-ci-97/xufang/model/Qwen3-4B_eagle3-ov-int4/";
        std::cout << "Testing model loading from: " << model_path << std::endl;
        std::cout << "Using draft model from: " << draft_model_path << std::endl;

        std::string lora_path_execution = "/home/openvino-ci-97/bell/speculative_decoding/Qwen3-4B-lora/adapter_model.safetensors";

        ov::genai::AdapterConfig adapter_config_execution;
        ov::genai::Adapter lora_execution(lora_path_execution);
        adapter_config_execution.add(lora_execution, 1.0f);

        std::cout << "Initializing LLM pipeline..." << std::endl;
        auto pipe_start = std::chrono::high_resolution_clock::now();

        ov::genai::LLMPipeline pipe(model_path,
                                   "GPU.1",
                                   ov::genai::adapters(adapter_config_execution),
                                   ov::genai::draft_model(draft_model_path, "GPU.1"));

        auto pipe_end = std::chrono::high_resolution_clock::now();
        auto pipe_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pipe_end - pipe_start).count();
        std::cout << "LLM Pipeline initialization took: " << pipe_duration << " ms" << std::endl;

        std::cout << "Model loaded successfully!" << std::endl;

        ov::genai::GenerationConfig config;
        config.max_new_tokens = 100;
        config.temperature = 1.0f;
        config.num_assistant_tokens = 4;

        pipe.start_chat();

        std::string prompt = "What is OpenVINO?";

        /*int warmup_runs = 3;
        std::cout << "\nUsing fixed prompt for warmup and test:\n" << prompt << std::endl;
        std::cout << "\nRunning " << warmup_runs << " warmup runs..." << std::endl;

        for (int i = 0; i < warmup_runs; ++i) {
            std::cout << "\nWarmup run " << (i + 1) << "..." << std::endl;
            auto warmup_streamer = [](std::string) {
                return ov::genai::StreamingStatus::RUNNING;
            };
            pipe.generate(prompt,
                          ov::genai::generation_config(config),
                          ov::genai::streamer(warmup_streamer),
                          ov::genai::adapters(adapter_config_execution));
            std::cout << std::endl;
        }*/

        std::cout << "\nStarting measured test run (SPD + execution-world LoRA)..." << std::endl;
        std::cout << "AI: ";
        auto streamer = [](std::string word) {
            std::cout << word << std::flush;
            return ov::genai::StreamingStatus::RUNNING;
        };

        auto result = pipe.generate(prompt,
                                    ov::genai::generation_config(config),
                                    ov::genai::streamer(streamer),
                                    ov::genai::adapters(adapter_config_execution));
        std::cout << std::endl;
        auto sd_perf_metrics = std::dynamic_pointer_cast<ov::genai::SDPerModelsPerfMetrics>(result.extended_perf_metrics);
        if (sd_perf_metrics) {
            auto main_model_metrics = sd_perf_metrics->main_model_metrics;
            std::cout << "\nMAIN MODEL " << std::endl;
            std::cout << "  Generate time: " << main_model_metrics.get_generate_duration().mean << " ms" << std::endl;
            std::cout << "  TTFT: " << main_model_metrics.get_ttft().mean  << " ± " << main_model_metrics.get_ttft().std << " ms" << std::endl;
            std::cout << "  TTST: " << main_model_metrics.get_ttst().mean  << " ± " << main_model_metrics.get_ttst().std << " ms/token " << std::endl;
            std::cout << "  TPOT: " << main_model_metrics.get_tpot().mean  << " ± " << main_model_metrics.get_tpot().std << " ms/iteration " << std::endl;
            std::cout << "  AVG Latency: " << main_model_metrics.get_latency().mean  << " ± " << main_model_metrics.get_latency().std << " ms/token " << std::endl;
            std::cout << "  Num generated token: " << main_model_metrics.get_num_generated_tokens() << " tokens" << std::endl;
            std::cout << "  Total iteration number: " << main_model_metrics.raw_metrics.m_durations.size() << std::endl;
            std::cout << "  Num accepted token: " << sd_perf_metrics->get_num_accepted_tokens() << " tokens" << std::endl;

            auto draft_model_metrics = sd_perf_metrics->draft_model_metrics;
            std::cout << "\nDRAFT MODEL " << std::endl;
            std::cout << "  Generate time: " << draft_model_metrics.get_generate_duration().mean << " ms" << std::endl;
            std::cout << "  TTFT: " << draft_model_metrics.get_ttft().mean  << " ms" << std::endl;
            std::cout << "  TTST: " << draft_model_metrics.get_ttst().mean  << " ms/token " << std::endl;
            std::cout << "  TPOT: " << draft_model_metrics.get_tpot().mean  << " ± " << draft_model_metrics.get_tpot().std << " ms/token " << std::endl;
            std::cout << "  AVG Latency: " << draft_model_metrics.get_latency().mean  << " ± " << draft_model_metrics.get_latency().std << " ms/iteration " << std::endl;
            std::cout << "  Num generated token: " << draft_model_metrics.get_num_generated_tokens() << " tokens" << std::endl;
            std::cout << "  Total iteration number: " << draft_model_metrics.raw_metrics.m_durations.size() << std::endl;
        }
        auto perf_metrics = result.perf_metrics;
        const auto gen = perf_metrics.get_generate_duration();
        const auto ttft = perf_metrics.get_ttft();
        const auto tpot = perf_metrics.get_tpot();
        const auto throughput = perf_metrics.get_throughput();
        const auto num_tokens = perf_metrics.get_num_generated_tokens();
        const auto num_input_tokens = perf_metrics.get_num_input_tokens();
        const double tps_excl_first = throughput_excluding_first_token(perf_metrics);

        std::cout << "\nPerformance metrics (SPD + execution-world LoRA):"
                    << "\nGenerate time: " << gen.mean << " ms"
                    << "\nTTFT: " << ttft.mean << " ± " << ttft.std << " ms"
                    << "\nTPOT: " << tpot.mean << " ± " << tpot.std << " ms/token"
                    << "\nThroughput: " << throughput.mean << " ± " << throughput.std << " tokens/s"
                    << "\nThroughput (exclude first token): " << tps_excl_first << " tokens/s"
                    << "\nNum generated token: " << num_tokens << " tokens"
                    << "\nNum input tokens: " << num_input_tokens << " tokens"
                    << std::endl;

        std::cout << "\nTest completed successfully!" << std::endl;
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}