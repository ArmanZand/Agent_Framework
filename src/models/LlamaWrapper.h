#ifndef AGENT_FRAMEWORK_LLAMAWRAPPER_H
#define AGENT_FRAMEWORK_LLAMAWRAPPER_H

#include <windows.h>
#include <string>
#include "llama.h"
#include <optional>

class LlamaWrapper {
public:
    ~LlamaWrapper();
    LlamaWrapper() = default;
    LlamaWrapper(LlamaWrapper && other) noexcept;
    LlamaWrapper& operator=(LlamaWrapper&& other) noexcept;

    LlamaWrapper(const LlamaWrapper&) = delete;
    LlamaWrapper& operator=(const LlamaWrapper&) = delete;

    static std::optional<LlamaWrapper> initialize();

    const std::string & get_error() const;

    //LLAMA EXPOSED FUNCTIONS
    //Log
    void (*log_set)(ggml_log_callback, void *) = nullptr;
    void (*ggml_abort)(const char *, int, const char *, ...) = nullptr;

    //Backend
    void (*backend_init)() = nullptr;
    void (*backend_free)() = nullptr;
    void (*ggml_backend_load_all)() = nullptr;

    //Model
    llama_context_params (*context_default_params)() = nullptr;
    llama_model_params (*model_default_params)() = nullptr;
    llama_model *(*model_load_from_file)(const char*, llama_model_params) = nullptr;
    llama_vocab *(*model_get_vocab)(const struct llama_model*) = nullptr;
    llama_context *(*init_from_model)(llama_model *, llama_context_params) = nullptr;
    uint32_t (*n_ctx)(const llama_context *) = nullptr;
    int32_t (*model_n_embd)(const llama_model *) = nullptr;
    int32_t (*model_n_embd_out)(const llama_model *) = nullptr;
    float * (*get_embeddings)(llama_context * ctx) = nullptr;
    float * (*get_embeddings_seq)(llama_context *, llama_seq_id) = nullptr;
    float * (*get_embeddings_ith)(llama_context *, int32_t) = nullptr;
    bool (*model_has_encoder)(const llama_model *) = nullptr;
    bool (*model_has_decoder)(const llama_model *) = nullptr;
    int32_t (*model_n_ctx_train)(const llama_model *) = nullptr;
    enum llama_pooling_type (*pooling_type)(const llama_context *) = nullptr;
    //Sampler
    llama_sampler_chain_params  (*sampler_chain_default_params)() = nullptr;
    llama_sampler *(*sampler_chain_init)(llama_sampler_chain_params) = nullptr;
    void (*sampler_chain_add)(llama_sampler*, llama_sampler*) = nullptr;
    llama_sampler *(*sampler_init_grammar)(const struct llama_vocab *, const char *, const char *) = nullptr;
    void (*sampler_reset)(struct llama_sampler *) = nullptr;
    llama_sampler *(*sampler_init_top_p)(float p, size_t) = nullptr;
    llama_sampler *(*sampler_init_min_p)(float, size_t) = nullptr;
    llama_sampler *(*sampler_init_temp)(float) = nullptr;
    llama_sampler *(*sampler_init_dist)(uint32_t) = nullptr;
    llama_sampler *(*sampler_init_top_k)(uint32_t) = nullptr;
    //Memory
    bool (*memory_seq_rm)(llama_memory_t, llama_seq_id, llama_pos, llama_pos) = nullptr;
    llama_pos (*memory_seq_pos_max)(llama_memory_t, llama_seq_id) = nullptr;
    llama_memory_t (*get_memory)(const llama_context *) = nullptr;
    void (*memory_clear)(llama_memory_t, bool) = nullptr;
    //Prompt
    int32_t (*tokenize)(const llama_vocab *, const char *, int32_t, llama_token *, int32_t, bool, bool) = nullptr;
    const char * (*model_chat_template)(const llama_model *, const char *) = nullptr;
    int32_t  (*chat_apply_template)(const char*, const llama_chat_message *, size_t, bool, char *, int32_t) = nullptr;
    llama_batch (*batch_get_one)(llama_token *, int32_t) = nullptr;
    llama_batch (*batch_init)(int32_t, int32_t, int32_t) = nullptr;
    int32_t (*decode)(llama_context *, llama_batch) = nullptr;
    llama_token (*sampler_sample)(llama_sampler *, llama_context *, int32_t) = nullptr;
    bool (*vocab_is_eog)(const llama_vocab *, llama_token) = nullptr;
    llama_token (*vocab_eos)(const llama_vocab *) = nullptr;
    llama_token (*vocab_bos)(const llama_vocab *) = nullptr;
    int32_t (*token_to_piece)(const llama_vocab *, llama_token, char *, int32_t, int32_t, bool) = nullptr;
    //Free
    void (*sampler_free)(llama_sampler *) = nullptr;
    void (*model_free)(llama_model*) = nullptr;
    void (*llama_free)(llama_context *) = nullptr;
    void (*batch_free)(llama_batch) = nullptr;
private:
    HMODULE llama_dll = nullptr;
    HMODULE ggml_dll = nullptr;
    HMODULE ggml_base_dll = nullptr;
    HMODULE ggml_cuda_dll = nullptr;
    std::string last_error;

    llama_model * model = nullptr;

    void cleanup();
    bool load_dlls();
    bool get_pointer_addresses();
};


#endif //AGENT_FRAMEWORK_LLAMAWRAPPER_H