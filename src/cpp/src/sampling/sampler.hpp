
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <list>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <map>
#include <algorithm>
#include <cmath>
#include <random>
#include <set>

#include "openvino/runtime/tensor.hpp"

#include "sampling/logit_transformers.hpp"
#include "sampling/logit_processor.hpp"
#include "continuous_batching/scheduler.hpp"
#include "sequence_group.hpp"
#include "threadpool.hpp"
#include "sampling/structured_output/structured_output_controller.hpp"

namespace ov::genai {
// Handle stop_token_ids
inline bool is_stop_token_id_hit(int64_t generated_token, const std::set<int64_t> & stop_token_ids) {
    for (auto & stop_token_id : stop_token_ids) {
        if (generated_token == stop_token_id)
            return true;
    }
    return false;
}

inline bool is_stop_token_id_hit_in_sequence_group(SequenceGroup::Ptr sequence_group, const std::set<int64_t>& stop_token_ids) {
    for (auto& sequence : sequence_group->get_running_sequences()) {
        const TokenIds& generated_tokens = sequence->get_generated_ids();
        if (!generated_tokens.empty() && is_stop_token_id_hit(generated_tokens.back(), stop_token_ids)) {
            return true;
        }
    }
    return false;
}

std::vector<Token> log_softmax(const ov::Tensor& logits, size_t batch_idx);

struct SamplerOutput {
    // IDs of sequences that need to be dropped
    std::vector<uint64_t> m_dropped_sequences;
    // IDs of sequences that need to be forked (note, the same sequence can be forked multiple times)
    // it will later be used by scheduler to fork block_tables for child sequences
    std::unordered_map<uint64_t, std::list<uint64_t>> m_forked_sequences;
    // store number of generated_tokens
    size_t num_generated_tokens = 0;
};

struct AssistingPipelineInfo {
    size_t max_removed_tokens_per_request = 0; 
    size_t min_generated_len = std::numeric_limits<size_t>::max();
    size_t updated_validation_len = 0;
};

struct SequenceGroupSamplingInfo {
    SamplerOutput sampler_output;
    AssistingPipelineInfo assisting_pipeline_info;

    AssistingPipelineInfo& get_assisting_pipeline_info() {
        return assisting_pipeline_info;
    }
};

class Sampler {
    class Searcher;
    class TreeSearcher;
    class GroupBeamSearcher;

    Logits _get_logit_vector(ov::Tensor logits, size_t batch_idx, size_t token_idx);
    Token _greedy_sample(const Logits& logits, size_t top_logprobs) const;
    std::vector<Token> _multinomial_sample(const Logits& logits, size_t num_tokens_per_sequence);
    std::vector<int64_t> _try_finish_generation(SequenceGroup::Ptr & sequence_group);

    bool validate_candidate(Sequence::Ptr running_sequence, size_t& token_idx, Token& sampled_token,
                            bool& is_extend_sequence, size_t& max_removed_tokens, bool do_sample, bool has_real_probolities);

    // Validate tree results from the target model using retrieve_indices and logits
    // Returns the number of valid tokens, and truncates sequence if mismatch is found
    size_t validate_tree_candidates(Sequence::Ptr& running_sequence, const ov::Tensor& sequence_group_logits, LogitProcessor& logit_processor, size_t num_tokens_to_validate);

    SequenceGroupSamplingInfo sample_from_sequence_group(SequenceGroup::Ptr sequence_group, ov::Tensor sequence_group_logits,
                                                        LogitProcessor& logit_processor, const std::pair<size_t, std::set<std::string>>& stop_strings,
                                                        bool is_validation_mode_enabled);

    std::mutex m_search_info_mutex;
    // request ID => beam search tracking information
    std::map<uint64_t, GroupBeamSearcher> m_beam_search_info;

    // request ID => tree search tracking information
    std::map<uint64_t, std::shared_ptr<TreeSearcher>> m_tree_search_info;
    std::mt19937 rng_engine;
    size_t seed = rng_engine.default_seed;
    // { request_id, logit_processor }
    std::map<uint64_t, LogitProcessor> m_logit_processors;
    // { request_id, { max_encoded_len, { stop_strings }}}
    std::map<int64_t, std::pair<size_t, std::set<std::string>>> m_stop_strings;

    Tokenizer m_tokenizer;

    ThreadPool m_thread_pool;
    std::shared_ptr<ov::op::v0::Constant> m_d2t_mapping; // Tensor to store draft_id_to_target_id mapping for eagle model, adding offsets to draft tokens after sampling
public:
    Sampler(const Sampler& rhs) = delete;
    Sampler(Sampler&& rhs) = delete;
    Sampler(size_t num_threads = 1): m_thread_pool(num_threads) {};
    explicit Sampler(const Tokenizer & tokenizer, size_t num_threads = 1) : m_tokenizer(tokenizer), m_thread_pool(num_threads) {};

    SamplerOutput sample(const std::vector<SequenceGroup::Ptr> & sequence_groups, ov::Tensor logits, bool is_validation_mode_enabled = false);
    void set_seed(size_t new_seed) {
        rng_engine.seed(new_seed);
        seed = new_seed;
    }
    size_t get_seed() { return seed; }

    void set_tokenizer(const Tokenizer& tokenizer) {
        m_tokenizer = tokenizer;
    }

    void clear_request_info(uint64_t request_id);

    LogitProcessor& get_logit_processor(uint64_t request_id);
    void create_logit_processor(uint64_t request_id, const GenerationConfig& sampling_parameters, const TokenIds& prompt);

    std::map<size_t, int32_t> get_beam_idxs(SequenceGroup::CPtr sequence_group);
    // pair with map with backend name and corresponding compiler init time, and vector of compile times for each concrete grammar
    std::pair<std::map<std::string, float>, std::vector<float>> get_structured_output_times();
    void clear_structured_output_compile_times();

    void set_d2t_for_decoding(const std::shared_ptr<ov::op::v0::Constant>& d2t) {
        m_d2t_mapping = d2t;
    };
};

class Sampler::Searcher {
public:
    struct Beam {
        Sequence::Ptr m_sequence;
        size_t m_global_beam_idx = 0;

        // beam is made on top of sequence
        float m_log_prob = 0.0f;
        int64_t m_token_id = -1;

        // cumulative log probabilities
        float m_score = -std::numeric_limits<float>::infinity();

        Beam(Sequence::Ptr sequence)
            : m_sequence(std::move(sequence)) { }

        size_t get_generated_len() const {
            return m_sequence->get_generated_len();
        }
    };

    static bool greater(const Beam& left, const Beam& right) {
        return left.m_score > right.m_score;
    }

protected:
    SequenceGroup::Ptr m_sequence_group;
    ov::genai::GenerationConfig m_parameters;
    Tokenizer m_tokenizer;

    explicit Searcher(SequenceGroup::Ptr sequence_group, Tokenizer tokenizer)
        : m_sequence_group(sequence_group), 
          m_parameters{m_sequence_group->get_sampling_parameters()},
          m_tokenizer(tokenizer) {
    }
};

class Sampler::TreeSearcher : public Sampler::Searcher {
    void tree_reset(SequenceGroup::Ptr& sequence_group);
    struct CandidateNode {
        int64_t token_id = -1;
        float score = -std::numeric_limits<float>::infinity();
        int tree_layer = 0;
        std::vector<std::shared_ptr<CandidateNode>> children;
        std::weak_ptr<CandidateNode> parent;

        CandidateNode() = default;
        CandidateNode(int64_t token_id, float score, int tree_layer)
            : token_id(token_id), score(score), tree_layer(tree_layer) {}
    };

    class TreeCandidateGraph {
    public:
        using NodePtr = std::shared_ptr<CandidateNode>;

        TreeCandidateGraph(const Beam& root_beam, int total_tokens = 0, int depth = 0)
            : total_tokens(total_tokens),
              max_depth(depth),
              current_depth(0) {
            root = std::make_shared<CandidateNode>(root_beam.m_token_id, root_beam.m_score, 0);
            layer_to_nodes.resize(1);
            layer_to_nodes[0].push_back(root);
        }

        NodePtr get_root() const {
            return root;
        }

        NodePtr add_candidate(const Beam& beam, const NodePtr& parent_node) {
            OPENVINO_ASSERT(parent_node != nullptr, "Parent node is null in candidate graph");

            const int next_layer = parent_node->tree_layer + 1;
            if (next_layer > max_depth) {
                return nullptr;
            }

            auto new_node = std::make_shared<CandidateNode>(beam.m_token_id, beam.m_score, next_layer);
            new_node->parent = parent_node;
            parent_node->children.push_back(new_node);

            if (layer_to_nodes.size() <= static_cast<size_t>(next_layer)) {
                layer_to_nodes.resize(static_cast<size_t>(next_layer) + 1);
            }
            layer_to_nodes[static_cast<size_t>(next_layer)].push_back(new_node);

            current_depth = std::max(current_depth, next_layer);
            return new_node;
        }
        bool is_ancestor(const NodePtr& ancestor, const NodePtr& node) const {
            if (!ancestor || !node) {
                return false;
            }

            auto cur = node;
            const CandidateNode* ancestor_raw = ancestor.get();
            while (cur) {
                if (cur.get() == ancestor_raw) {
                    return true;
                }
                cur = cur->parent.lock();
            }
            return false;
        }
        std::vector<NodePtr> get_top_k_candidates() const {
            if (total_tokens <= 0)
                return {};

            // Use min-heap to efficiently get top-k candidates (excluding root)
            auto cmp = [](const NodePtr& a, const NodePtr& b) {
                return a->score > b->score;  // min-heap
            };

            std::priority_queue<NodePtr,
                                std::vector<NodePtr>,
                                decltype(cmp)>
                min_heap(cmp);

            // BFS traversal to find all candidates (excluding root)
            std::queue<NodePtr> bfs_queue;
            bfs_queue.push(root);

            while (!bfs_queue.empty()) {
                auto node = bfs_queue.front();
                bfs_queue.pop();

                if (node != root) {
                    if (min_heap.size() < static_cast<size_t>(total_tokens)) {
                        min_heap.push(node);
                    } else if (node->score > min_heap.top()->score) {
                        min_heap.pop();
                        min_heap.push(node);
                    }
                }

                for (const auto& child : node->children) {
                    bfs_queue.push(child);
                }
            }

            std::vector<NodePtr> result;
            result.reserve(min_heap.size() + 1);

            result.push_back(root);

            while (!min_heap.empty()) {
                result.push_back(min_heap.top());
                min_heap.pop();
            }

            std::sort(result.begin(), result.end(), [](const NodePtr& a, const NodePtr& b) {
                return a->score > b->score;
            });

            return result;
        }
        std::vector<NodePtr> get_current_layer_candidates() const {
            if (current_depth < 0 || layer_to_nodes.size() <= static_cast<size_t>(current_depth)) {
                return {};
            }
            return layer_to_nodes[static_cast<size_t>(current_depth)];
        }

        std::vector<NodePtr> get_leaf_nodes_from_candidates(const std::vector<NodePtr>& candidates) const {
            std::vector<NodePtr> leaf_nodes;
            std::unordered_set<const CandidateNode*> candidate_ids;

            // Build set of candidate node IDs
            for (const auto& node : candidates) {
                candidate_ids.insert(node.get());
            }

            // Check each candidate to see if it's a leaf in the selected set
            for (const auto& candidate : candidates) {
                // Check if this node has any children in the candidate set
                bool has_candidate_child = false;
                for (const auto& child : candidate->children) {
                    if (candidate_ids.count(child.get()) > 0) {
                        has_candidate_child = true;
                        break;
                    }
                }

                if (!has_candidate_child) {
                    leaf_nodes.push_back(candidate);
                }
            }

            return leaf_nodes;
        }

        std::vector<NodePtr> get_path_to_node(const NodePtr& node) const {
            std::vector<NodePtr> path;
            auto current = node;
            while (current) {
                path.push_back(current);
                current = current->parent.lock();
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

    private:
        NodePtr root;
        std::vector<std::vector<NodePtr>> layer_to_nodes;

        int total_tokens;
        int max_depth;
        int current_depth;
    };
    size_t m_tree_layer_counter = 0;
    size_t m_past_generate_len = 0;
    std::shared_ptr<TreeCandidateGraph> m_tree_candidate_graph;
    std::vector<Beam> m_beams;
    std::vector<TreeCandidateGraph::NodePtr> m_beam_nodes;
    uint64_t m_org_group_id = 0;
    int64_t* m_d2t; // Draft-to-target token ID offset
public:
    explicit TreeSearcher(SequenceGroup::Ptr sequence_group, ov::Tensor d2t);
    bool is_depth_reached() const {
        return m_tree_layer_counter >= m_parameters.tree_params.tree_depth;
    }
    void select_top_k(const ov::Tensor& logits, SamplerOutput& sampler_output, LogitProcessor& logit_processor);
    void finalize_tree(SamplerOutput& sampler_output, LogitProcessor& logit_processor);
};

class Sampler::GroupBeamSearcher : public Sampler::Searcher {
    using Sampler::Searcher::Beam;
    struct Group {
        std::vector<Beam> ongoing;  // Best beams in front
        std::vector<Beam> min_heap;  // The worst of the best completed beams is the first
        bool done = false;

        int64_t finish(Beam beam, const ov::genai::GenerationConfig& sampling_params);
        void is_done();
    };

    std::vector<Group> m_groups;
public:
    explicit GroupBeamSearcher(SequenceGroup::Ptr sequence_group, Tokenizer tokenizer);

    void select_next_tokens(const ov::Tensor& logits, SamplerOutput& sampler_output, const std::pair<size_t, std::set<std::string>>& stop_strings);
    void finalize(SamplerOutput& sampler_output);
    std::map<size_t, int32_t> get_beam_idxs();
};
}
