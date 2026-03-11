// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <future>

#include "fast_draft_strategy.hpp"
#include "openvino/op/constant.hpp"
#include "update_request_structs.hpp"
namespace ov::genai {
class KVUpdateWrapper {
public:
    KVUpdateWrapper() = default;
    explicit KVUpdateWrapper(const ov::genai::ModelDesc& kv_model_desc, const size_t& kv_num);
    ~KVUpdateWrapper() = default;
    void update_request(uint64_t request_id, const GeneratedSequences& candidates);
    void infer(const ov::Tensor& block_indices,
                          const ov::Tensor& block_indices_begins,
                          const ov::Tensor& block_update_indices,
                          const ov::Tensor& block_update_indices_begins,
                          const std::vector<ov::Tensor>& key_caches,
                          const std::vector<ov::Tensor>& value_caches) {

                          };
    size_t m_kv_layers_num = 0;
    ov::CompiledModel m_kv_update_model;
    mutable ov::InferRequest m_request;
    ov::Tensor m_block_update_indices_tensor, m_block_update_indices_begins_tensor, m_block_indices_tensor, m_block_indices_begins_tensor;
};
class ContinuousBatchingPipeline::Eagle3DecodingImpl : public ContinuousBatchingPipeline::SpeculativeDecodingImpl {
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

    Eagle3DecodingImpl(const ov::genai::ModelDesc& main_model_desc, const ov::genai::ModelDesc& draft_model_desc, const std::vector<int>& hidden_layers_to_abstract);

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer,
             const std::optional<std::vector<ov::Tensor>>& token_type_ids = std::nullopt,
             const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids = std::nullopt,
             const std::optional<std::vector<ov::Tensor>>& prompt_ids = std::nullopt) override;
    void step() override;
    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> token_type_ids = std::nullopt,
                                 std::optional<ov::Tensor> prompt_ids = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;
protected:
    void update_eagle_pipeline_params(const std::shared_ptr<ov::op::v0::Constant>& d2t_tensor);
    // Creates draft model input by removing the first token from the original input sequence.
    ov::Tensor create_draft_input_ids(const ov::Tensor& original_input_ids);
    std::shared_ptr<KVUpdateWrapper> m_kv_update_wrapper;
};
}
