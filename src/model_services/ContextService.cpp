#include "ContextService.h"

#include "BackendService.h"

ContextService::ContextService(std::shared_ptr<ModelService> model_service, const ContextConfig & config) : model_service_(std::move(model_service)), config_(config) {
}


ContextService::~ContextService() {
    if (context_ && BackendService::instance().is_running()) {
        BackendService::instance().llama().llama_free(context_);
        if (config_.verbose) {
            fprintf(stdout, "[ContextService] Unloaded context.\n");
        }
    }
}

bool ContextService::initialize() {
    if (!BackendService::instance().is_running()) {
        fprintf(stderr,"[ContextService] Cannot initialize. Backend service is not running.\n");
        fflush(stderr);
        return false;
    }

    auto & llama  = BackendService::instance().llama();
    llama_context_params ctx_params = llama.context_default_params();
    ctx_params.n_ctx = config_.context_size;
    ctx_params.n_batch = config_.batch_size;
    ctx_params.n_threads = config_.threads;
    ctx_params.n_threads_batch = config_.threads_batch;
    ctx_params.embeddings = config_.generate_embeddings;

    if (!allocate_context(ctx_params)) {
        return false;
    }

   pooling_type_ = llama.pooling_type(context_);


    return true;
}

int ContextService::token_count() const {
    if (!context_) return {};
    auto& llama = BackendService::instance().llama();
    llama_memory_t mem = llama.get_memory(context_);
    if (!mem) return 0;
    llama_pos pos = llama.memory_seq_pos_max(mem, 0);
    return (pos < 0) ? 0 : static_cast<int>(pos + 1);
}

std::vector<float> ContextService::get_sequence_embedding(llama_seq_id seq) const {
    if (!context_) return {};
    if (pooling_type_ == LLAMA_POOLING_TYPE_NONE) return {};

    auto & llama = BackendService::instance().llama();
    int dim = model_service_->n_embd_out();
    if (dim <= 0) return {};

    float * ptr = llama.get_embeddings_seq(context_, seq);
    if (!ptr) return {};
    return std::vector<float>(ptr, ptr + dim);
}

std::vector<float> ContextService::get_token_embedding(int batch_index) const {
    if (!context_) return {};
    auto & llama = BackendService::instance().llama();
    int dim = model_service_->n_embd_out();
    if (dim <= 0) return {};
    float * ptr = llama.get_embeddings_ith(context_, batch_index);
    if (!ptr) return {};
    return std::vector<float>(ptr, ptr + dim);
}

std::vector<float> ContextService::get_all_embeddings() const {
    if (!context_) return {};
    auto & llama = BackendService::instance().llama();
    int dim = model_service_->n_embd_out();
    if (dim <= 0) return {};
    float * ptr = llama.get_embeddings(context_);
    if (!ptr) return {};
    int n = token_count();
    return std::vector<float>(ptr, ptr + static_cast<size_t>(n) * dim);
}

void ContextService::clear_kv_cache(bool zero_data) {
    if (!context_) return;
    auto & llama = BackendService::instance().llama();
    llama_memory_t mem = llama.get_memory(context_);
    if (mem) llama.memory_clear(mem, zero_data);
}

llama_pos ContextService::max_seq_pos(llama_seq_id seq) const {
    if (!context_) return -1;
    auto & llama = BackendService::instance().llama();
    llama_memory_t mem = llama.get_memory(context_);
    if (!mem) return -1;
    return llama.memory_seq_pos_max(mem, seq);
}

bool ContextService::allocate_context(const llama_context_params & ctx_params) {
    auto & llama  = BackendService::instance().llama();
    if (config_.verbose) {
        fprintf(stdout, "[ContextService] Allocating context - ");
        fflush(stdout);
    }
    context_ = llama.init_from_model(model_service_->model(), ctx_params);
    allocated_context_ = llama.n_ctx(context_);
    if (!context_) {
        fprintf(stderr, "[ContextService] Failed to allocate model context.\n");
        return false;
    }
    if (config_.verbose) {
        fprintf(stdout, "DONE\n");
        fflush(stdout);
    }
    return true;
}


