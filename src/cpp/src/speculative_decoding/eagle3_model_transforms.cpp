// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eagle3_model_transforms.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "json_utils.hpp"
#include "logger.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace eagle3 {

Eagle3RTInfo extract_eagle3_info_from_config(ov::AnyMap& config, const std::filesystem::path& models_path) {
    Eagle3RTInfo eagle_rt_info;
    if (config.find("eagle3_mode") != config.end()) {
        eagle_rt_info.eagle3_mode = config.at("eagle3_mode").as<bool>();
        config.erase("eagle3_mode");
        auto it = config.find("hidden_layers_list");
        if (it != config.end()) {
            OPENVINO_ASSERT(it->second.is<std::vector<int32_t>>(),
                            "hidden_layers_list must be a vector of int32_t values");
            eagle_rt_info.hidden_layers_list = it->second.as<std::vector<int32_t>>();
            config.erase("hidden_layers_list");
        } else {
            // compute the layers from number of hidden layers
            auto config_file_path = models_path / "config.json";
            OPENVINO_ASSERT(std::filesystem::exists(config_file_path), "Cannot deduce layers for hidden layer extraction because the file is missing: ", config_file_path);
            std::ifstream file(config_file_path);

            nlohmann::json data = nlohmann::json::parse(file);
            using ov::genai::utils::read_json_param;
            int num_decoder_layers = 0;
            read_json_param(data, "num_hidden_layers", num_decoder_layers);

            // Ensure sufficient layers for meaningful feature extraction
            // Minimum of 10 layers is based on practical LLM architectures (e.g., small GPT-2 has 12 layers)
            OPENVINO_ASSERT(
                num_decoder_layers >= 10,
                "num_decoder_layers must be at least 10 for automatic hidden layer selection, got: ",
                num_decoder_layers,
                ". For models with fewer layers, please explicitly specify 'hidden_layers_list' in the configuration.");

            // The following default hidden layer selection corresponds to the EAGLE reference implementation:
            // https://github.com/SafeAILab/EAGLE/blob/0ea94696/eagle/model/modeling_llama_kv.py#L1138
            // These layers (2, num_decoder_layers / 2, num_decoder_layers - 3) are chosen to capture features from
            // early, middle, and late stages of the decoder, as recommended by the EAGLE authors.
            // Note: Integer division (num_decoder_layers / 2) is intentional and produces the desired behavior
            // for typical LLM layer counts (e.g., 12→6, 24→12, 32→16).
            // If you wish to use different layers, provide the "hidden_layers_list" parameter in the config.
            eagle_rt_info.hidden_layers_list = { 2, num_decoder_layers / 2, num_decoder_layers - 3 };
        }
        OPENVINO_ASSERT(eagle_rt_info.hidden_layers_list.size() == 3, "Eagle3 is expected to provide exactly three layers for extraction");
    }
    return eagle_rt_info;
}

void apply_eagle3_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties) {
    if (model->has_rt_info("eagle3_mode") && model->get_rt_info<bool>("eagle3_mode")) {
        properties["eagle3_mode"] = true;
        if (model->has_rt_info("hidden_layers_list")) {
            properties["hidden_layers_list"] = model->get_rt_info<std::vector<int>>("hidden_layers_list");
        }
    }
}

void share_vocabulary(const std::shared_ptr<ov::Model>& main_model, const std::shared_ptr<ov::Model>& draft_model) {
    // extract embedding weight from main model
    auto find_embedding_gather = [](const std::shared_ptr<ov::Model>& model)
        -> std::shared_ptr<ov::Node> {
        constexpr size_t MIN_VOCAB_SIZE_THRESHOLD = 1000;
        for (const auto& node : model->get_ordered_ops()) {
            auto gather = std::dynamic_pointer_cast<ov::op::util::GatherBase>(node);
            if (!gather) continue;
            // [vocab, hidden_size] * [batch, seq_len] -> [batch, seq_len, hidden_size]
            auto data_node = gather->input_value(0).get_node_shared_ptr();
            auto indices_node = gather->input_value(1).get_node_shared_ptr();
            if (!data_node || !indices_node) continue;
            // indices_node should be on parameter path, maybe this is better rule
            ov::PartialShape ps = data_node->get_output_partial_shape(0);
            if (ps.rank().is_static() && ps.rank().get_length() >= 2) {
                if (ps[0].is_static() && ps[0].get_length() > MIN_VOCAB_SIZE_THRESHOLD) { // Heuristic: vocab size > 1000
                    return gather;
                }
            }
            std::string fname = data_node->get_friendly_name();
            if (fname.find("embed_tokens") != std::string::npos ||
                fname.find("embedding") != std::string::npos) {
                return gather;
            }
        }
        return nullptr;
    };
    auto main_gather  = find_embedding_gather(main_model);
    auto draft_gather = find_embedding_gather(draft_model);
    if (!main_gather || !draft_gather) {
        return;
    }
    auto main_weight_node = main_gather->input_value(0).get_node_shared_ptr();
    auto draft_weight_node = draft_gather->input_value(0).get_node_shared_ptr();

    if (main_weight_node.get() == draft_weight_node.get()) {
        return;
    }

    GENAI_INFO("Copying embedding weights from main to draft model for eagle3 speculative decoding.");

    // Helper function to recursively clone a node and its inputs
    // This handles cases where embedding has intermediate ops (Convert, FakeQuantize, etc.)
    std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&,
                                            std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>>&)>
        clone_node_recursive =
            [&](const std::shared_ptr<ov::Node>& node,
                std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>>& cloned_nodes) -> std::shared_ptr<ov::Node> {

        auto it = cloned_nodes.find(node.get());
        if (it != cloned_nodes.end()) {
            return it->second;
        }

        std::shared_ptr<ov::Node> cloned;

        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            // For Constant nodes, create a deep copy with new data
            cloned = std::make_shared<ov::op::v0::Constant>(constant->get_element_type(),
                                                            constant->get_shape(),
                                                            constant->get_data_ptr());
        } else {
            // For other nodes, clone recursively with cloned inputs
            ov::OutputVector cloned_inputs;
            for (size_t i = 0; i < node->get_input_size(); ++i) {
                auto input_node = node->get_input_node_shared_ptr(i);
                auto cloned_input = clone_node_recursive(input_node, cloned_nodes);
                cloned_inputs.push_back(cloned_input->output(node->get_input_source_output(i).get_index()));
            }
            cloned = node->clone_with_new_inputs(cloned_inputs);
        }

        cloned->set_friendly_name(node->get_friendly_name() + "_cloned_for_draft");
        cloned_nodes[node.get()] = cloned;
        return cloned;
    };

    // Clone the entire subgraph from main model
    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> cloned_nodes;
    auto cloned_weight_node = clone_node_recursive(main_weight_node, cloned_nodes);

    OPENVINO_ASSERT(cloned_weight_node,
                    "Failed to clone embedding weight node from main model to draft model. "
                    "This is required for Eagle3 speculative decoding.");

    // Replace draft model's weight node with the cloned subgraph
    // This avoids cross-model references by duplicating the vocabulary weights
    draft_weight_node->output(0).replace(cloned_weight_node->output(0));
}

void move_fc_from_draft_to_main(std::shared_ptr<ov::Model>& draft_model, std::shared_ptr<ov::Model>& main_model) {
    // extract the FC transform weight from draft model
    auto remove_fc_and_rewire = [](const std::shared_ptr<ov::Model>& model) -> std::shared_ptr<ov::Node> {
        for (const auto& node : model->get_ordered_ops()) {
            auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(node);
            if (!matmul_node) continue;
            auto input_node = matmul_node->get_input_node_shared_ptr(0);
            auto param_node = ov::as_type_ptr<ov::op::v0::Parameter>(input_node);
            if (!param_node || (input_node->get_friendly_name().find("hidden_states") == std::string::npos && input_node->get_friendly_name().find("target_hidden") == std::string::npos)) continue;
            // Rewire all outputs of this MatMul to use the input_node directly
            for (auto& output : matmul_node->outputs()) {
                for (auto& target : output.get_target_inputs()) {
                    target.replace_source_output(input_node);
                }
            }
            return matmul_node->input_value(1).get_node_shared_ptr();
        }
        return nullptr;
    };
    auto fc_weights = remove_fc_and_rewire(draft_model);
    if (!fc_weights)
        OPENVINO_THROW("Failed to locate FC weights in eagle3 draft model for shifting to main model.");
    // now we create the fc into main model
    for (const auto& result : main_model->get_results()) {
        auto input_node = result->input_value(0).get_node_shared_ptr();
        if (input_node && input_node->get_friendly_name().find("eagle3_hidden_states_concat") != std::string::npos) {
            auto matmul = std::make_shared<ov::op::v0::MatMul>(input_node, fc_weights, false, true);
            matmul->set_friendly_name("eagle3_hidden_state_fc");
            result->input(0).replace_source_output(matmul);
            break;
        }
    }
}

// Helper function to find d2t result node in the model
static std::shared_ptr<ov::op::v0::Result> find_d2t_result_node(const std::shared_ptr<ov::Model>& model) {
    for (const auto& result : model->get_results()) {
        auto input_node = result->input_value(0).get_node_shared_ptr();
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(input_node);
        if (constant && constant->get_friendly_name().find("d2t") != std::string::npos) {
            return result;
        }
    }
    return nullptr;
}

std::shared_ptr<ov::op::v0::Constant> extract_d2t_mapping_table(const std::shared_ptr<ov::Model>& model) {
    // extract result nodes from model
    auto d2t_result = find_d2t_result_node(model);
    if (d2t_result) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(d2t_result->input_value(0).get_node_shared_ptr());
        model->remove_result(d2t_result);
        model->validate_nodes_and_infer_types();
        return constant;
    }
    return nullptr;
}

void transform_hidden_state(std::shared_ptr<ov::Model>& model, const std::vector<int32_t>& hidden_layers_to_abstract) {
    if (hidden_layers_to_abstract.empty()) {
        return;
    }
    OPENVINO_ASSERT(
        hidden_layers_to_abstract.size() == 3 || hidden_layers_to_abstract.size() == 1 || hidden_layers_to_abstract.size() == 5,
        "Expected exactly 1 or 3 hidden layers for extraction: 1 for draft model, 3 for main model (early/middle/late stages)."
    );

    std::vector<std::string> patterns;
    if (hidden_layers_to_abstract.size() > 1) {
        patterns.reserve(hidden_layers_to_abstract.size());
        for (int32_t idx : hidden_layers_to_abstract) {
            patterns.emplace_back("layers." + std::to_string(idx) + "/"); // main description
        }
    } else {
        patterns.emplace_back("midlayer"); // draft description
    }

    // Helper: check if node is a residual Add node with expected structure
    auto is_residual_node = [](const std::shared_ptr<ov::Node>& node) -> bool {
        if (const auto& add = ov::as_type_ptr<ov::op::v1::Add>(node)) {
            auto input1 = add->get_input_node_shared_ptr(1);
            auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(input1);
            if (!matmul) return false;
            auto matmul_input = matmul->get_input_node_shared_ptr(0);
            return matmul_input && ov::is_type<ov::op::v1::Multiply>(matmul_input);
        }
        return false;
    };

    std::vector<ov::Output<ov::Node>> residual_outputs;
    for (const auto& node : model->get_ordered_ops()) {
        if (!is_residual_node(node)) continue;
        const std::string& name = node->get_friendly_name();
        for (const auto& pattern : patterns) {
            if (name.find(pattern) != std::string::npos) {
                residual_outputs.push_back(node->output(0));
                break;
            }
        }
    }

    if (!residual_outputs.empty()) {
        OPENVINO_ASSERT(residual_outputs.size() == patterns.size(),
                        "Number of extracted hidden states does not match the requested number.");
        std::shared_ptr<ov::Node> node_to_operate;
        if (residual_outputs.size() > 1) {
            auto concat = std::make_shared<ov::op::v0::Concat>(residual_outputs, -1);
            concat->set_friendly_name("eagle3_hidden_states_concat");
            node_to_operate = concat;
        } else {
            node_to_operate = residual_outputs[0].get_node_shared_ptr();
        }
        auto result = std::make_shared<ov::op::v0::Result>(node_to_operate);
        const std::string output_name = "last_hidden_state";
        result->output(0).set_names({output_name});
        result->set_friendly_name(output_name);
        // NPUW use this info to identify manually added outputs
        result->get_rt_info()["manually_added_output"] = true;
        model->add_results({result});
    }
}

ov::Tensor slice_hidden_state_for_last_token(const ov::Tensor& hidden_features) {
    OPENVINO_ASSERT(hidden_features.get_size() > 0, "Hidden features tensor is empty");

    const auto shape = hidden_features.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[1] > 0, "Expected shape [1, seq_len, hidden_size]");

    const size_t seq_len = shape[1];

    auto [start_coord, end_coord] = ov::genai::utils::make_roi(shape, 1, seq_len - 1, seq_len);
    return ov::Tensor(hidden_features, start_coord, end_coord);
}

std::shared_ptr<ov::Model> construct_eagle3_kv_update_model(const std::shared_ptr<ov::Model>& main_model) {
    // the kv update model acceptes all kv cache inputs from main_model
    // extra inputs for updating kv cache: block_indices, block_indices_begins, block_update_indices, block_update_indices_begins， all with element::i32, PartialShape{-1}
    // the output is the updated kv cache with same shape and element type as main model's kv cache
    auto kv_update_model = std::make_shared<ov::Model>(main_model->get_results(), main_model->get_parameters(), "eagle3_kv_update_model");
    using namespace ov;
    ParameterVector inputs;
    // clone the kv cache parameters from the main model
    auto params = main_model->get_parameters();
    std::vector<Output<Node>> key_caches;
    std::vector<Output<Node>> value_caches;
    for (const auto& param : params) {
        const std::string& name = param->get_friendly_name();
        // Find paged_attention op connected to this param
        std::shared_ptr<ov::Node> paged_attention_op = nullptr;
        for (const auto& node : main_model->get_ordered_ops()) {
            // Typical paged_attention op is custom, so check op type and input
            if (node->get_friendly_name().find("PagedAttentionExtension") != std::string::npos) {
                for (size_t idx = 0; idx < node->get_input_size(); ++idx) {
                    if (node->get_input_node_shared_ptr(idx).get() == param.get()) {
                        paged_attention_op = node;
                        break;
                    }
                }
                if (paged_attention_op) break;
            }
        }
        if (name.find("key_cache") != std::string::npos) {
            auto cloned_param = std::make_shared<ov::op::v0::Parameter>(param->get_element_type(), param->get_partial_shape());
            cloned_param->set_friendly_name(name);
            cloned_param->output(0).set_names({name});
            // Clone runtime info from paged_attention op if found
            if (paged_attention_op) {
                for (const auto& [key, value] : paged_attention_op->get_rt_info()) {
                    cloned_param->get_rt_info()[key] = value;
                }
            }
            inputs.push_back(cloned_param);
            key_caches.push_back(cloned_param);
        } else if (name.find("value_cache") != std::string::npos) {
            auto cloned_param = std::make_shared<ov::op::v0::Parameter>(param->get_element_type(), param->get_partial_shape());
            cloned_param->set_friendly_name(name);
            cloned_param->output(0).set_names({name});
            // Clone runtime info from paged_attention op if found
            if (paged_attention_op) {
                for (const auto& [key, value] : paged_attention_op->get_rt_info()) {
                    cloned_param->get_rt_info()[key] = value;
                }
            }
            inputs.push_back(cloned_param);
            value_caches.push_back(cloned_param);
        }
    }

    auto block_indices_begins = std::make_shared<op::v0::Parameter>(
        element::i32, PartialShape{-1});
    block_indices_begins->set_friendly_name("block_indices_begins");
    block_indices_begins->output(0).set_names({"block_indices_begins"});
    inputs.push_back(block_indices_begins);

    auto block_indices = std::make_shared<op::v0::Parameter>(
        element::i32, PartialShape{-1});
    block_indices->set_friendly_name("block_indices");
    block_indices->output(0).set_names({"block_indices"});
    inputs.push_back(block_indices);

    auto block_update_indices = std::make_shared<op::v0::Parameter>(
        element::i32, PartialShape{-1});
    block_update_indices->set_friendly_name("block_update_indices");
    block_update_indices->output(0).set_names({"block_update_indices"});
    inputs.push_back(block_update_indices);

    auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(
        element::i32, PartialShape{-1});
    block_update_indices_begins->set_friendly_name("block_update_indices_begins");
    block_update_indices_begins->output(0).set_names({"block_update_indices_begins"});
    inputs.push_back(block_update_indices_begins);

    ResultVector results;
    size_t pair_count = std::min(key_caches.size(), value_caches.size());
    for (size_t i = 0; i < pair_count; ++i) {
        auto key_gather = std::make_shared<op::v8::Gather>(
            key_caches[i], block_update_indices, std::make_shared<op::v0::Constant>(element::i32, ov::Shape{1}, -1));
        key_gather->set_friendly_name("reordered_key_cache_" + std::to_string(i));
        auto key_scatter = std::make_shared<op::v3::ScatterUpdate>(
            key_caches[i], block_indices, key_gather, std::make_shared<op::v0::Constant>(element::i32, ov::Shape{1}, -1));
        key_scatter->set_friendly_name("updated_key_cache_" + std::to_string(i));

        auto value_gather = std::make_shared<op::v8::Gather>(
            value_caches[i], block_update_indices, std::make_shared<op::v0::Constant>(element::i32, ov::Shape{1}, -2));
        value_gather->set_friendly_name("reordered_value_cache_" + std::to_string(i));
        auto value_scatter = std::make_shared<op::v3::ScatterUpdate>(
            value_caches[i], block_indices, value_gather, std::make_shared<op::v0::Constant>(element::i32, ov::Shape{1}, -2));
        value_scatter->set_friendly_name("updated_value_cache_" + std::to_string(i));

        // Concat key and value scatter outputs along last axis
        auto concat = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{key_scatter->output(0), value_scatter->output(0)}, -1);
        concat->set_friendly_name("kv_cache_pair_concat_" + std::to_string(i));
        results.push_back(std::make_shared<op::v0::Result>(concat));
    }

    auto model = std::make_shared<Model>(results, inputs, "kv_cache_reorder_model");
    // addition runtime info for identification
    model->get_rt_info()["auxiliary_kv_update_model"] = true;
    return model;
}
}  // namespace eagle3
namespace dflash {
// read from local config, to 
DFlashRTInfo extract_dflash_info_from_config(const std::filesystem::path& config) {
    DFlashRTInfo dflash_rt_info;
    // check if config is exist, if not return default dflash_rt_info with empty hidden_layers_list and mask_token_id = -1
    if (!config.empty() && std::filesystem::exists(config)) {
        std::ifstream file(config);
        nlohmann::json data = nlohmann::json::parse(file);
        using ov::genai::utils::read_json_param;
        /*  sample of dflash config:
        "dflash_config": {
            "mask_token_id": 151669,
            "target_layer_ids": [
            1,
            9,
            17,
            25,
            33
            ]
        },*/
        read_json_param(data, "dflash_config.mask_token_id", dflash_rt_info.mask_token_id);
        read_json_param(data, "dflash_config.target_layer_ids", dflash_rt_info.hidden_layers_list);
    }

    return dflash_rt_info;
}

void share_lm_head_weights(const std::shared_ptr<ov::Model>& main_model, const std::shared_ptr<ov::Model>& draft_model) {
    // Extract lm_head weight producer from model.
    // Prefer MatMul-based topology (e.g. __module.lm_head/aten::linear/MatMul),
    // keep Gather as fallback for other model variants.
    auto find_lm_head_weight = [](const std::shared_ptr<ov::Model>& model)
        -> std::shared_ptr<ov::Node> {
        for (const auto& node : model->get_ordered_ops()) {
            auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(node);
            if (matmul && matmul->get_friendly_name().find("lm_head") != std::string::npos) {
                return matmul->input_value(1).get_node_shared_ptr();
            }

            auto gather = ov::as_type_ptr<ov::op::util::GatherBase>(node);
            if (!gather)
                continue;
            auto data_node = gather->input_value(0).get_node_shared_ptr();
            if (data_node && data_node->get_friendly_name().find("lm_head") != std::string::npos) {
                return data_node;
            }
        }
        return nullptr;
    };
    auto main_weight_node = find_lm_head_weight(main_model);
    auto draft_weight_node = find_lm_head_weight(draft_model);
    if (!main_weight_node || !draft_weight_node) {
        return;
    }

    if (main_weight_node.get() == draft_weight_node.get()) {
        return;
    }

    GENAI_INFO("Copying LM head weights from main to draft model for D-Flash speculative decoding.");

    // Clone the full producer path (including quantization/dequantization ops),
    // not only the terminal Constant node.
    std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&,
                                            std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>>&)> 
        clone_node_recursive =
            [&](const std::shared_ptr<ov::Node>& node,
                std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>>& cloned_nodes) -> std::shared_ptr<ov::Node> {

        auto it = cloned_nodes.find(node.get());
        if (it != cloned_nodes.end()) {
            return it->second;
        }

        std::shared_ptr<ov::Node> cloned;
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            cloned = std::make_shared<ov::op::v0::Constant>(constant->get_element_type(),
                                                            constant->get_shape(),
                                                            constant->get_data_ptr());
        } else {
            ov::OutputVector cloned_inputs;
            for (size_t i = 0; i < node->get_input_size(); ++i) {
                auto input_node = node->get_input_node_shared_ptr(i);
                auto cloned_input = clone_node_recursive(input_node, cloned_nodes);
                cloned_inputs.push_back(cloned_input->output(node->get_input_source_output(i).get_index()));
            }
            cloned = node->clone_with_new_inputs(cloned_inputs);
        }

        cloned->set_friendly_name(node->get_friendly_name() + "_cloned_for_draft");
        cloned_nodes[node.get()] = cloned;
        return cloned;
    };

    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> cloned_nodes;
    auto cloned_weight_node = clone_node_recursive(main_weight_node, cloned_nodes);
    OPENVINO_ASSERT(cloned_weight_node,
                    "Failed to clone LM head weight node path from main model to draft model.");

    draft_weight_node->output(0).replace(cloned_weight_node->output(0));
}

void normalize_draft_kvproj_concat_axis(std::shared_ptr<ov::Model>& draft_model) {
    size_t updated_concat_count = 0;
    size_t updated_reshape_count = 0;
    for (const auto& node : draft_model->get_ordered_ops()) {
        auto concat = ov::as_type_ptr<ov::op::v0::Concat>(node);
        if (!concat)
            continue;

        const auto& name = concat->get_friendly_name();
        if (name.find("self_attn/aten::cat") == std::string::npos)
            continue;

        // Only rewrite problematic k_proj/v_proj concat path: both inputs are MatMul and
        // concat is authored along sequence axis (1) for non-flattened [B, S, H].
        // With flattened [B*S, 1, H], runtime concatenation should happen on axis 0.
        if (concat->get_input_size() != 2 || concat->get_axis() != 1)
            continue;

        // Try to get MatMul nodes, possibly through Reshape
        auto get_matmul = [](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::op::v0::MatMul> {
            // First try direct MatMul
            auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(node);
            if (matmul)
                return matmul;

            // If not, try through Reshape
            auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(node);
            if (reshape) {
                return ov::as_type_ptr<ov::op::v0::MatMul>(reshape->get_input_node_shared_ptr(0));
            }

            return nullptr;
        };

        auto lhs = get_matmul(concat->get_input_node_shared_ptr(0));
        auto rhs = get_matmul(concat->get_input_node_shared_ptr(1));
        if (!lhs || !rhs)
            continue;

        const auto& lhs_name = lhs->get_friendly_name();
        const auto& rhs_name = rhs->get_friendly_name();
        const bool is_kv_proj_pair =
            ((lhs_name.find("self_attn.k_proj/") != std::string::npos && rhs_name.find("self_attn.k_proj/") != std::string::npos) ||
             (lhs_name.find("self_attn.v_proj/") != std::string::npos && rhs_name.find("self_attn.v_proj/") != std::string::npos));
        if (!is_kv_proj_pair)
            continue;

        concat->set_axis(0);

        // Keep downstream reshape math consistent with flattened [B*S, 1, H] semantics.
        // Original shape formulas are built for axis=1 concatenation over [B, S, H].
        for (const auto& target_input : concat->output(0).get_target_inputs()) {
            if (target_input.get_index() != 0)
                continue;

            auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(target_input.get_node()->shared_from_this());
            if (!reshape)
                continue;

            auto concat_ps = concat->get_output_partial_shape(0);
            auto reshape_ps = reshape->get_output_partial_shape(0);
            if (concat_ps.rank().is_dynamic() || reshape_ps.rank().is_dynamic() ||
                concat_ps.rank().get_length() != 3 || reshape_ps.rank().get_length() != 4 ||
                !concat_ps[2].is_static() || !reshape_ps[3].is_static()) {
                continue;
            }

            const int64_t hidden = concat_ps[2].get_length();
            const int64_t head_size = reshape_ps[3].get_length();
            if (head_size <= 0 || hidden <= 0 || (hidden % head_size != 0))
                continue;

            const int64_t num_heads = hidden / head_size;
            auto concat_shape = std::make_shared<ov::op::v3::ShapeOf>(concat);
            auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
            auto idx0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            auto tokens = std::make_shared<ov::op::v8::Gather>(concat_shape, idx0, axis0);
            auto one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
            auto nh = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_heads});
            auto hs = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {head_size});
            auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{tokens, one, nh, hs}, 0);
            reshape->input(1).replace_source_output(new_shape);
            updated_reshape_count++;
        }

        updated_concat_count++;
    }

    if (updated_concat_count > 0) {
        GENAI_INFO(
            "Adjusted {} draft self-attn k_proj/v_proj Concat node(s) axis from 1 to 0 and updated {} dependent Reshape node(s).",
            updated_concat_count,
            updated_reshape_count);
        draft_model->validate_nodes_and_infer_types();
    }
}

}  // namespace dflash

void change_draft_sdpa(std::shared_ptr<ov::Model>& draft_model) {
    using namespace ov::op;

    size_t modified_sdpa_count = 0;
    size_t removed_reshape_count = 0;
    size_t removed_unsqueeze_count = 0;

    // Iterate through all nodes to find ScaledDotProductAttention operations
    for (const auto& node : draft_model->get_ops()) {
        auto sdpa = std::dynamic_pointer_cast<v13::ScaledDotProductAttention>(node);
        if (!sdpa) {
            continue;
        }

        // Input 0: query - remove Reshape if exists
        // Input 1: key - remove Unsqueeze if exists
        // Input 2: value - remove Unsqueeze if exists

        bool modified = false;

        // Handle input 0 (query) - remove Reshape
        if (sdpa->get_input_size() > 0) {
            auto query_source = sdpa->input_value(0);
            auto reshape_node = std::dynamic_pointer_cast<v1::Reshape>(query_source.get_node_shared_ptr());

            if (reshape_node) {
                // Replace SDPA's query input with Reshape's input
                auto reshape_input = reshape_node->input_value(0);
                sdpa->input(0).replace_source_output(reshape_input);
                removed_reshape_count++;
                modified = true;
            }
        }

        // Handle input 1 (key) - remove Unsqueeze
        if (sdpa->get_input_size() > 1) {
            auto key_source = sdpa->input_value(1);
            auto unsqueeze_node = std::dynamic_pointer_cast<v0::Unsqueeze>(key_source.get_node_shared_ptr());

            if (unsqueeze_node) {
                // Replace SDPA's key input with Unsqueeze's input
                auto unsqueeze_input = unsqueeze_node->input_value(0);
                sdpa->input(1).replace_source_output(unsqueeze_input);
                removed_unsqueeze_count++;
                modified = true;
            }
        }

        // Handle input 2 (value) - remove Unsqueeze
        if (sdpa->get_input_size() > 2) {
            auto value_source = sdpa->input_value(2);
            auto unsqueeze_node = std::dynamic_pointer_cast<v0::Unsqueeze>(value_source.get_node_shared_ptr());

            if (unsqueeze_node) {
                // Replace SDPA's value input with Unsqueeze's input
                auto unsqueeze_input = unsqueeze_node->input_value(0);
                sdpa->input(2).replace_source_output(unsqueeze_input);
                removed_unsqueeze_count++;
                modified = true;
            }
        }

        if (modified) {
            modified_sdpa_count++;
        }
    }

    if (modified_sdpa_count > 0) {
        GENAI_INFO(
            "Modified {} SDPA node(s) in draft model: removed {} Reshape and {} Unsqueeze node(s).",
            modified_sdpa_count,
            removed_reshape_count,
            removed_unsqueeze_count);
        draft_model->validate_nodes_and_infer_types();
    }
}

void update_positional_encoding_to_relative(std::shared_ptr<ov::Model>& draft_model) {
    using namespace ov::op;

    // Step 1: Find the two Multiply nodes
    std::shared_ptr<ov::Node> hidden_norm_multiply = nullptr;
    std::shared_ptr<ov::Node> input_layernorm_multiply = nullptr;

    for (const auto& node : draft_model->get_ops()) {
        const auto& name = node->get_friendly_name();

        if (name.find("__module.model.hidden_norm/aten::mul/Multiply_1") != std::string::npos) {
            hidden_norm_multiply = node;
        } else if (name.find("__module.model.layers.0.input_layernorm/aten::mul/Multiply_1") != std::string::npos) {
            input_layernorm_multiply = node;
        }
    }

    if (!hidden_norm_multiply || !input_layernorm_multiply) {
        GENAI_INFO("update_positional_encoding_to_relative: Required Multiply nodes not found");
        return;
    }

    // Step 2: Create Concat node combining the two outputs
    // Concat along axis 1 (seq_len dimension)
    auto concat_node = std::make_shared<v0::Concat>(
        ov::OutputVector{hidden_norm_multiply->output(0), input_layernorm_multiply->output(0)},
        0  // axis = 1
    );
    concat_node->set_friendly_name("positional_encoding_concat");

    // Step 3: Find the rotary_emb MatMul node
    std::shared_ptr<v0::MatMul> rotary_matmul = nullptr;
    for (const auto& node : draft_model->get_ops()) {
        const auto& name = node->get_friendly_name();
        if (name.find("__module.model.rotary_emb/aten::matmul/MatMul") != std::string::npos) {
            rotary_matmul = std::dynamic_pointer_cast<v0::MatMul>(node);
            if (rotary_matmul) {
                break;
            }
        }
    }

    if (!rotary_matmul) {
        GENAI_INFO("update_positional_encoding_to_relative: rotary_emb MatMul not found");
        return;
    }

    // Step 4: Recursively traverse up from MatMul to find ShapeOf node
    std::function<std::shared_ptr<v3::ShapeOf>(std::shared_ptr<ov::Node>)> find_shapeof =
    [&](std::shared_ptr<ov::Node> current) -> std::shared_ptr<v3::ShapeOf> {
        // Check if current node is ShapeOf
        auto shapeof = std::dynamic_pointer_cast<v3::ShapeOf>(current);
        if (shapeof) {
            return shapeof;
        }

        // Recursively search up the chain through all inputs
        for (size_t i = 0; i < current->get_input_size(); i++) {
            auto input_node = current->get_input_node_shared_ptr(i);
            auto found = find_shapeof(input_node);
            if (found) {
                return found;
            }
        }
        return nullptr;
    };

    auto old_shapeof_node = find_shapeof(rotary_matmul);

    if (old_shapeof_node) {
        // Create a new ShapeOf node connected to concat
        auto new_shapeof = std::make_shared<v3::ShapeOf>(concat_node->output(0), old_shapeof_node->get_output_element_type(0));
        new_shapeof->set_friendly_name("positional_encoding_shapeof");

        // Replace all usages of old_shapeof that are in the MatMul chain
        // We need to find which consumer of old_shapeof leads to rotary_matmul
        for (auto& target_input : old_shapeof_node->output(0).get_target_inputs()) {
            auto consumer = target_input.get_node()->shared_from_this();

            // Check if this consumer is in the path to rotary_matmul
            std::unordered_set<ov::Node*> visited;
            std::function<bool(std::shared_ptr<ov::Node>)> is_in_matmul_chain =
            [&](std::shared_ptr<ov::Node> node) -> bool {
                if (node == rotary_matmul) {
                    return true;
                }

                // Avoid infinite recursion
                if (visited.count(node.get()) > 0) {
                    return false;
                }
                visited.insert(node.get());

                // Check all consumers of this node
                for (auto& out_input : node->output(0).get_target_inputs()) {
                    if (is_in_matmul_chain(out_input.get_node()->shared_from_this())) {
                        return true;
                    }
                }
                return false;
            };

            if (is_in_matmul_chain(consumer)) {
                // Replace this usage with new ShapeOf
                target_input.replace_source_output(new_shapeof->output(0));
                GENAI_INFO("update_positional_encoding_to_relative: Replaced ShapeOf usage in '{}' with new concat-based ShapeOf",
                          consumer->get_friendly_name());
            }
        }

        draft_model->validate_nodes_and_infer_types();
        return;
    }

    GENAI_INFO("update_positional_encoding_to_relative: Could not find ShapeOf node in chain");
}

void update_slice_axis_update(std::shared_ptr<ov::Model>& draft_model) {
    using namespace ov::op;

    // Step 1: Find the Subtract node
    std::shared_ptr<ov::Node> subtract_node = nullptr;
    for (const auto& node : draft_model->get_ops()) {
        const auto& name = node->get_friendly_name();
        if (name.find("__module.model.layers.0.self_attn/aten::sub/Subtract") != std::string::npos) {
            subtract_node = node;
            break;
        }
    }

    if (!subtract_node) {
        std::cout << "update_slice_axis_update: Subtract node not found" << std::endl;
        return;
    }

    std::cout << "update_slice_axis_update: Found Subtract node: " << subtract_node->get_friendly_name() << std::endl;

    // Step 2: Find the two Gather inputs
    std::vector<std::shared_ptr<ov::op::v8::Gather>> gather_nodes;

    for (size_t i = 0; i < subtract_node->get_input_size(); i++) {
        auto input_node = subtract_node->get_input_node_shared_ptr(i);

        // Try to find Gather node (might be direct or through intermediate nodes)
        std::function<std::shared_ptr<v8::Gather>(std::shared_ptr<ov::Node>, int)> find_gather;
        find_gather = [&](std::shared_ptr<ov::Node> node, int depth) -> std::shared_ptr<v8::Gather> {
            if (depth > 5) return nullptr; // Limit search depth

            auto gather = std::dynamic_pointer_cast<v8::Gather>(node);
            if (gather) {
                return gather;
            }

            // Search through inputs
            for (size_t j = 0; j < node->get_input_size(); j++) {
                auto result = find_gather(node->get_input_node_shared_ptr(j), depth + 1);
                if (result) return result;
            }

            return nullptr;
        };

        auto gather = find_gather(input_node, 0);
        if (gather) {
            gather_nodes.push_back(gather);
        }
    }

    if (gather_nodes.empty()) {
        std::cout << "update_slice_axis_update: No Gather nodes found in Subtract inputs" << std::endl;
        return;
    }

    std::cout << "update_slice_axis_update: Found " << gather_nodes.size() << " Gather node(s)" << std::endl;

    // Step 3: Dump Gather indices and axis values, then modify them
    size_t modified_count = 0;
    for (size_t i = 0; i < gather_nodes.size(); i++) {
        auto gather = gather_nodes[i];
        std::cout << "  Gather " << i << ": " << gather->get_friendly_name() << std::endl;

        // Get current axis
        int64_t old_axis = 0;
        if (gather->get_input_size() > 2) {
            auto axis_node = std::dynamic_pointer_cast<v0::Constant>(gather->get_input_node_shared_ptr(2));
            if (axis_node) {
                auto axis_data = axis_node->cast_vector<int64_t>();
                old_axis = axis_data.empty() ? 0 : axis_data[0];
                std::cout << "    Old Axis: " << old_axis << std::endl;
            } else {
                std::cout << "    Axis: dynamic (not constant)" << std::endl;
            }
        }

        // Get current indices
        auto indices_node = std::dynamic_pointer_cast<v0::Constant>(gather->get_input_node_shared_ptr(1));
        if (indices_node) {
            auto indices_shape = indices_node->get_shape();
            std::cout << "    Indices shape: [";
            for (size_t j = 0; j < indices_shape.size(); j++) {
                if (j > 0) std::cout << ", ";
                std::cout << indices_shape[j];
            }
            std::cout << "]" << std::endl;

            // Try to get indices values (limit output)
            if (indices_node->get_element_type() == ov::element::i64) {
                auto indices_data = indices_node->cast_vector<int64_t>();
                if (indices_data.size() <= 10) {
                    std::cout << "    Old Indices values: [";
                    for (size_t j = 0; j < indices_data.size(); j++) {
                        if (j > 0) std::cout << ", ";
                        std::cout << indices_data[j];
                    }
                    std::cout << "]" << std::endl;
                } else {
                    std::cout << "    Old Indices values: [" << indices_data[0] << "...] (total " << indices_data.size() << " elements)" << std::endl;
                }
            } else if (indices_node->get_element_type() == ov::element::i32) {
                auto indices_data = indices_node->cast_vector<int32_t>();
                if (indices_data.size() <= 10) {
                    std::cout << "    Old Indices values: [";
                    for (size_t j = 0; j < indices_data.size(); j++) {
                        if (j > 0) std::cout << ", ";
                        std::cout << indices_data[j];
                    }
                    std::cout << "]" << std::endl;
                } else {
                    std::cout << "    Old Indices values: [" << indices_data[0] << "...] (total " << indices_data.size() << " elements)" << std::endl;
                }
            }
        } else {
            std::cout << "    Indices: dynamic (not constant)" << std::endl;
        }

        // Get input data shape
        std::cout << "    Input 0 shape: " << gather->get_input_partial_shape(0) << std::endl;
        std::cout << "    Output shape: " << gather->get_output_partial_shape(0) << std::endl;

        // Step 4: Modify indices and axis to 0
        // Create new indices constant with value [0]
        auto new_indices = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        new_indices->set_friendly_name(gather->get_friendly_name() + "_new_indices");

        // Create new axis constant with value 0
        auto new_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        new_axis->set_friendly_name(gather->get_friendly_name() + "_new_axis");

        // Replace Gather's inputs
        gather->input(1).replace_source_output(new_indices->output(0));  // indices
        if (gather->get_input_size() > 2) {
            gather->input(2).replace_source_output(new_axis->output(0));  // axis
        }

        std::cout << "    Modified: indices=[0], axis=0" << std::endl;
        modified_count++;
    }

    if (modified_count > 0) {
        std::cout << "update_slice_axis_update: Modified " << modified_count << " Gather node(s)" << std::endl;
        draft_model->validate_nodes_and_infer_types();
    }
}

namespace {
// Helper function to dump Slice node inputs
void dump_slice_inputs(std::shared_ptr<ov::Node> slice_node, int slice_index) {
    using namespace ov::op;

    std::cout << "  Slice " << slice_index << " inputs:" << std::endl;
    std::cout << "    Total inputs: " << slice_node->get_input_size() << std::endl;

    // Slice typically has 3-5 inputs: data, start, stop, [step], [axes]
    for (size_t i = 0; i < slice_node->get_input_size(); i++) {
        std::string input_name;
        switch (i) {
            case 0: input_name = "data"; break;
            case 1: input_name = "start"; break;
            case 2: input_name = "stop"; break;
            case 3: input_name = "step"; break;
            case 4: input_name = "axes"; break;
            default: input_name = "input_" + std::to_string(i); break;
        }

        std::cout << "    Input " << i << " (" << input_name << "):" << std::endl;

        auto input_node = slice_node->get_input_node_shared_ptr(i);
        std::cout << "      Node type: " << input_node->get_type_name() << std::endl;
        std::cout << "      Shape: " << slice_node->get_input_partial_shape(i) << std::endl;

        // Try to get constant value
        auto constant_node = std::dynamic_pointer_cast<v0::Constant>(input_node);
        if (constant_node) {
            auto shape = constant_node->get_shape();
            std::cout << "      Constant shape: [";
            for (size_t j = 0; j < shape.size(); j++) {
                if (j > 0) std::cout << ", ";
                std::cout << shape[j];
            }
            std::cout << "]" << std::endl;

            // Try to print values
            if (constant_node->get_element_type() == ov::element::i64) {
                auto values = constant_node->cast_vector<int64_t>();
                if (values.size() <= 10) {
                    std::cout << "      Values (i64): [";
                    for (size_t j = 0; j < values.size(); j++) {
                        if (j > 0) std::cout << ", ";
                        std::cout << values[j];
                    }
                    std::cout << "]" << std::endl;
                } else {
                    std::cout << "      Values (i64): [" << values[0] << ", ...] (total " << values.size() << " elements)" << std::endl;
                }
            } else if (constant_node->get_element_type() == ov::element::i32) {
                auto values = constant_node->cast_vector<int32_t>();
                if (values.size() <= 10) {
                    std::cout << "      Values (i32): [";
                    for (size_t j = 0; j < values.size(); j++) {
                        if (j > 0) std::cout << ", ";
                        std::cout << values[j];
                    }
                    std::cout << "]" << std::endl;
                } else {
                    std::cout << "      Values (i32): [" << values[0] << ", ...] (total " << values.size() << " elements)" << std::endl;
                }
            }
        } else {
            std::cout << "      Dynamic (not constant)" << std::endl;
        }
    }

    std::cout << "    Output shape: " << slice_node->get_output_partial_shape(0) << std::endl;
}
}  // anonymous namespace

void update_strided_slice_axis_update(std::shared_ptr<ov::Model>& draft_model) {
    using namespace ov::op;

    // Step 1: Find the two Slice nodes
    std::shared_ptr<ov::Node> slice_node = nullptr;
    std::shared_ptr<ov::Node> slice_1_node = nullptr;

    for (const auto& node : draft_model->get_ops()) {
        const auto& name = node->get_friendly_name();

        if (name.find("__module.model.layers.0.self_attn/aten::narrow/Slice") != std::string::npos) {
            if (name.find("Slice_1") != std::string::npos) {
                slice_1_node = node;
            } else {
                // Make sure it's the base Slice, not Slice_1
                if (name.find("__module.model.layers.0.self_attn/aten::narrow/Slice") == name.rfind("__module.model.layers.0.self_attn/aten::narrow/Slice")) {
                    slice_node = node;
                }
            }
        }
    }

    std::vector<std::shared_ptr<ov::Node>> slice_nodes;
    if (slice_node) {
        slice_nodes.push_back(slice_node);
        std::cout << "update_strided_slice_axis_update: Found Slice node: " << slice_node->get_friendly_name() << std::endl;
        dump_slice_inputs(slice_node, 0);
    } else {
        std::cout << "update_strided_slice_axis_update: Slice node not found" << std::endl;
    }

    if (slice_1_node) {
        slice_nodes.push_back(slice_1_node);
        std::cout << "update_strided_slice_axis_update: Found Slice_1 node: " << slice_1_node->get_friendly_name() << std::endl;
        dump_slice_inputs(slice_1_node, 1);
    } else {
        std::cout << "update_strided_slice_axis_update: Slice_1 node not found" << std::endl;
    }

    // Step 2: Modify axes to 0
    size_t modified_count = 0;
    for (auto& slice : slice_nodes) {
        // Slice has input 4 as axes (if present)
        if (slice->get_input_size() > 4) {
            auto axes_node = slice->get_input_node_shared_ptr(4);
            auto axes_constant = std::dynamic_pointer_cast<v0::Constant>(axes_node);

            if (axes_constant) {
                // Get old axes value
                std::vector<int64_t> old_axes;
                if (axes_constant->get_element_type() == ov::element::i64) {
                    old_axes = axes_constant->cast_vector<int64_t>();
                } else if (axes_constant->get_element_type() == ov::element::i32) {
                    auto axes_i32 = axes_constant->cast_vector<int32_t>();
                    old_axes.assign(axes_i32.begin(), axes_i32.end());
                }

                std::cout << "  Modifying " << slice->get_friendly_name() << std::endl;
                std::cout << "    Old axes: [";
                for (size_t i = 0; i < old_axes.size(); i++) {
                    if (i > 0) std::cout << ", ";
                    std::cout << old_axes[i];
                }
                std::cout << "]" << std::endl;

                // Create new axes constant with value [0]
                auto new_axes = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
                new_axes->set_friendly_name(slice->get_friendly_name() + "_new_axes");

                // Replace Slice's axes input
                slice->input(4).replace_source_output(new_axes->output(0));

                std::cout << "    New axes: [0]" << std::endl;
                modified_count++;
            } else {
                std::cout << "  Warning: axes input is not a constant for " << slice->get_friendly_name() << std::endl;
            }
        } else {
            std::cout << "  Warning: Slice node " << slice->get_friendly_name() << " has no axes input (input count: " << slice->get_input_size() << ")" << std::endl;
        }
    }

    if (modified_count > 0) {
        std::cout << "update_strided_slice_axis_update: Modified " << modified_count << " Slice node(s)" << std::endl;
        draft_model->validate_nodes_and_infer_types();
    }
}

}  // namespace utils
}  // namespace genai
}  // namespace ov
