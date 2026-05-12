#ifndef AGENT_FRAMEWORK_CONTEXTSERVICE_H
#define AGENT_FRAMEWORK_CONTEXTSERVICE_H
#include <memory>
#include <vector>

#include "ModelService.h"

struct ContextConfig {
    int context_size = 8192;
    int batch_size = 512;
    int threads = 4;
    int threads_batch = 4;
    bool verbose = false;
    bool generate_embeddings = false;
};



class ContextService {
public:
    ContextService(
        std::shared_ptr<ModelService> model_service,
        const ContextConfig & config);
    ~ContextService();

    ContextService(const ContextService &) = delete;
    ContextService & operator=(const ContextService&) = delete;

    bool initialize();

    llama_context * context() const { return context_; }
    int allocated_context() const { return allocated_context_; }
    enum llama_pooling_type pooling_type() const { return pooling_type_; }

    int token_count() const;
    int remaining() const { return allocated_context_ - token_count(); }
    bool is_near_full(int threshold = 256) const { return remaining() < threshold; }

    int n_batch() const { return config_.batch_size; }
    std::vector<float> get_sequence_embedding(llama_seq_id seq = 0) const;
    std::vector<float> get_token_embedding(int batch_index) const;
    std::vector<float> get_all_embeddings() const;

    void clear_kv_cache(bool zero_data = false);
    llama_pos max_seq_pos(llama_seq_id seq = 0) const;
private:
    bool allocate_context(const llama_context_params & ctx_params);

    std::shared_ptr<ModelService> model_service_;
    ContextConfig config_;
    llama_context * context_ = nullptr;
    int allocated_context_;
    enum llama_pooling_type pooling_type_;
};


#endif //AGENT_FRAMEWORK_CONTEXTSERVICE_H