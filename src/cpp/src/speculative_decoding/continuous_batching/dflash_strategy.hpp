// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <future>

#include "eagle3_strategy.hpp"
#include "openvino/op/constant.hpp"
#include "update_request_structs.hpp"
namespace ov::genai {
class ContinuousBatchingPipeline::DFlashDecodingImpl : public ContinuousBatchingPipeline::SpeculativeDecodingImpl {
public:
    template<class Impl>
    friend std::vector<EncodedGenerationResult> generate_common(
        Impl*,
        const std::vector<ov::Tensor>&,
        const std::vector<GenerationConfig>&,
        const StreamerVariant&,
        std::optional<std::vector<ov::Tensor>>,
        std::optional<std::vector<ov::Tensor>>,
        GenerateStrategy&);

    DFlashDecodingImpl(const ov::genai::ModelDesc& main_model_desc, const ov::genai::ModelDesc& draft_model_desc, const std::vector<int>& hidden_layers_to_abstract);

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer,
             const std::optional<std::vector<ov::Tensor>>& token_type_ids = std::nullopt,
             const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids = std::nullopt,
             const std::optional<std::vector<ov::Tensor>>& prompt_ids = std::nullopt,
             const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list = std::nullopt) override;
    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> token_type_ids = std::nullopt,
                                 std::optional<ov::Tensor> prompt_ids = std::nullopt,
                                 std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;
    void step() override;
protected:
    void update_dflash_pipeline_params(const std::shared_ptr<ov::op::v0::Constant>& d2t_tensor);
    // Creates draft model input based on D-block size.
    ov::Tensor create_draft_input_ids(const ov::Tensor& original_input_ids);
};
}
