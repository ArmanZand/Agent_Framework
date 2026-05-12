#include "LlamaWrapper.h"
#include <iostream>

LlamaWrapper::LlamaWrapper(LlamaWrapper&& other) noexcept
    : llama_dll(other.llama_dll)
    , ggml_dll(other.ggml_dll)
    , ggml_base_dll(other.ggml_base_dll)
    , last_error(std::move(other.last_error))
    , log_set(other.log_set)
    , ggml_abort(other.ggml_abort)
    , ggml_backend_load_all(other.ggml_backend_load_all)
    , backend_init(other.backend_init)
    , backend_free(other.backend_free)
    , context_default_params(other.context_default_params)
    , model_default_params(other.model_default_params)
    , model_load_from_file(other.model_load_from_file)
    , model_get_vocab(other.model_get_vocab)
    , init_from_model(other.init_from_model)
    , n_ctx(other.n_ctx)
    , model_n_embd(other.model_n_embd)
    , model_n_embd_out(other.model_n_embd_out)
    , get_embeddings(other.get_embeddings)
    , get_embeddings_seq(other.get_embeddings_seq)
    , pooling_type(other.pooling_type)
    , model_has_encoder(other.model_has_encoder)
    , model_has_decoder(other.model_has_decoder)
    , model_n_ctx_train(other.model_n_ctx_train)
    , sampler_chain_default_params(other.sampler_chain_default_params)
    , sampler_chain_init(other.sampler_chain_init)
    , sampler_chain_add(other.sampler_chain_add)
    , sampler_init_grammar(other.sampler_init_grammar)
    , sampler_reset(other.sampler_reset)
    , sampler_init_top_p(other.sampler_init_top_p)
    , sampler_init_min_p(other.sampler_init_min_p)
    , sampler_init_temp(other.sampler_init_temp)
    , sampler_init_dist(other.sampler_init_dist)
    , sampler_init_top_k(other.sampler_init_top_k)
    , memory_seq_rm(other.memory_seq_rm)
    , memory_seq_pos_max(other.memory_seq_pos_max)
    , get_memory(other.get_memory)
    , memory_clear(other.memory_clear)
    , tokenize(other.tokenize)
    , model_chat_template(other.model_chat_template)
    , chat_apply_template(other.chat_apply_template)
    , batch_get_one(other.batch_get_one)
    , batch_init(other.batch_init)
    , decode(other.decode)
    , sampler_sample(other.sampler_sample)
    , vocab_is_eog(other.vocab_is_eog)
    , vocab_eos(other.vocab_eos)
    , vocab_bos(other.vocab_bos)
    , token_to_piece(other.token_to_piece)
    , sampler_free(other.sampler_free)
    , model_free(other.model_free)
    , llama_free(other.llama_free)
    , batch_free(other.batch_free)
{
    other.llama_dll = nullptr;
    other.ggml_dll = nullptr;
    other.ggml_base_dll = nullptr;
}

LlamaWrapper& LlamaWrapper::operator=(LlamaWrapper&& other) noexcept {
    if (this != &other) {
        cleanup();
        llama_dll = other.llama_dll;
        ggml_dll = other.ggml_dll;
        ggml_base_dll = other.ggml_base_dll;
        last_error = std::move(other.last_error);

        log_set = other.log_set;
        ggml_abort = other.ggml_abort;
        ggml_backend_load_all = other.ggml_backend_load_all;
        backend_init = other.backend_init;
        backend_free = other.backend_free;
        context_default_params = other.context_default_params;
        model_default_params = other.model_default_params;
        model_load_from_file = other.model_load_from_file;
        model_get_vocab = other.model_get_vocab;
        init_from_model = other.init_from_model;
        n_ctx = other.n_ctx;
        model_n_embd = other.model_n_embd;
        model_n_embd_out = other.model_n_embd_out;
        get_embeddings = other.get_embeddings;
        get_embeddings_seq = other.get_embeddings_seq;
        model_has_encoder = other.model_has_encoder;
        model_has_decoder = other.model_has_decoder;
        model_n_ctx_train = other.model_n_ctx_train;
        pooling_type = other.pooling_type;
        sampler_chain_default_params = other.sampler_chain_default_params;
        sampler_chain_init = other.sampler_chain_init;
        sampler_chain_add = other.sampler_chain_add;
        sampler_init_grammar = other.sampler_init_grammar;
        sampler_reset = other.sampler_reset;
        sampler_init_top_p = other.sampler_init_top_p;
        sampler_init_min_p = other.sampler_init_min_p;
        sampler_init_temp = other.sampler_init_temp;
        sampler_init_dist = other.sampler_init_dist;
        sampler_init_top_k = other.sampler_init_top_k;
        memory_seq_rm = other.memory_seq_rm;
        memory_seq_pos_max = other.memory_seq_pos_max;
        get_memory = other.get_memory;
        memory_clear = other.memory_clear;
        tokenize = other.tokenize;
        model_chat_template = other.model_chat_template;
        chat_apply_template = other.chat_apply_template;
        batch_get_one = other.batch_get_one;
        batch_init = other.batch_init;
        decode = other.decode;
        sampler_sample = other.sampler_sample;
        vocab_is_eog = other.vocab_is_eog;
        vocab_eos = other.vocab_eos;
        vocab_bos = other.vocab_bos;
        token_to_piece = other.token_to_piece;
        sampler_free = other.sampler_free;
        model_free = other.model_free;
        llama_free = other.llama_free;
        batch_free = other.batch_free;
        other.llama_dll = nullptr;
        other.ggml_dll = nullptr;
        other.ggml_base_dll = nullptr;
    }
    return *this;
}

LlamaWrapper::~LlamaWrapper() {
    cleanup();
}

std::optional<LlamaWrapper> LlamaWrapper::initialize() {
    LlamaWrapper llama_wrapper;
    if (!llama_wrapper.load_dlls())
        return std::nullopt;

    if (!llama_wrapper.get_pointer_addresses())
        return std::nullopt;

    return llama_wrapper;
}

void LlamaWrapper::cleanup() {
    if (llama_dll) {
        FreeLibrary(llama_dll);
        llama_dll = nullptr;
    }
    if (ggml_dll) {
        FreeLibrary(ggml_dll);
        ggml_dll = nullptr;
    }
    if (ggml_base_dll) {
        FreeLibrary(ggml_base_dll);
        ggml_base_dll = nullptr;
    }
}

bool LlamaWrapper::load_dlls() {
    char exe_path[MAX_PATH];
    GetModuleFileName(nullptr, exe_path, MAX_PATH);
    std::string working_directory = exe_path;
    working_directory = working_directory.substr(0, working_directory.find_last_of('\\'));

    //Load ggml.dll
    const std::string ggml_dll_path = working_directory + "\\ggml.dll";
    ggml_dll = LoadLibrary(ggml_dll_path.c_str());
    if (!ggml_dll) {
        const DWORD err_code = ::GetLastError();
        last_error = "Failed to load '" + ggml_dll_path + "'. Error Code: " + std::to_string(err_code);
        cleanup();
        return false;
    }

    //Load ggml-base.dll
    const std::string ggml_base_dll_path = working_directory + "\\ggml-base.dll";
    ggml_base_dll = LoadLibrary(ggml_base_dll_path.c_str());
    if (!ggml_base_dll) {
        const DWORD err_code = ::GetLastError();
        last_error = "Failed to load '" + ggml_base_dll_path + "'. Error Code: " + std::to_string(err_code);
        cleanup();
        return false;
    }

    //Load ggml-cuda.dll
    const std::string ggml_cuda_dll_path = working_directory + "\\ggml-cuda.dll";
    ggml_cuda_dll = LoadLibrary(ggml_cuda_dll_path.c_str());
    if (!ggml_cuda_dll) {
        const DWORD err_code = ::GetLastError();
        last_error = "Failed to load '" + ggml_cuda_dll_path + "'. Error Code: " + std::to_string(err_code);
        cleanup();
        return false;
    }

    //Load llama.dll
    const std::string llama_dll_path = working_directory + "\\llama.dll";
    llama_dll = LoadLibrary(llama_dll_path.c_str());
    if (!llama_dll) {
        const DWORD err_code = ::GetLastError();
        last_error = "Failed to load '" + llama_dll_path + "'. Error Code: " + std::to_string(err_code);
        cleanup();
        return false;
    }


    return true;
}

bool LlamaWrapper::get_pointer_addresses() {
    //Log
    log_set = reinterpret_cast<void (*)(ggml_log_callback, void*)>(GetProcAddress(llama_dll, "llama_log_set"));
    ggml_abort = reinterpret_cast<void (*)(const char *, int, const char *,...)>(GetProcAddress(ggml_base_dll, "ggml_abort"));
    //Backend Load
    ggml_backend_load_all = reinterpret_cast<void (*)()>(GetProcAddress(ggml_dll, "ggml_backend_load_all"));
    backend_init = reinterpret_cast<void (*)()>(GetProcAddress(llama_dll, "llama_backend_init"));
    backend_free = reinterpret_cast<void (*)()>(GetProcAddress(llama_dll, "llama_backend_free"));
    //Model
    context_default_params = reinterpret_cast<llama_context_params(*)()>(GetProcAddress(llama_dll, "llama_context_default_params"));
    model_default_params = reinterpret_cast<llama_model_params(*)()>(GetProcAddress(llama_dll, "llama_model_default_params"));
    model_load_from_file = reinterpret_cast<llama_model*(*)(const char * , llama_model_params)>(GetProcAddress(llama_dll, "llama_model_load_from_file"));
    model_get_vocab = reinterpret_cast<llama_vocab*(*)(const llama_model*)>(GetProcAddress(llama_dll, "llama_model_get_vocab"));
    init_from_model = reinterpret_cast<llama_context*(*)(llama_model *, llama_context_params)>(GetProcAddress(llama_dll, "llama_init_from_model"));
    n_ctx = reinterpret_cast<uint32_t(*)(const llama_context *)>(GetProcAddress(llama_dll, "llama_n_ctx"));
    model_n_embd = reinterpret_cast<int32_t(*)(const llama_model *)>(GetProcAddress(llama_dll, "llama_model_n_embd"));
    model_n_embd_out = reinterpret_cast<int32_t(*)(const llama_model *)>(GetProcAddress(llama_dll, "llama_model_n_embd_out"));
    get_embeddings = reinterpret_cast<float*(*)(llama_context *)>(GetProcAddress(llama_dll, "llama_get_embeddings"));
    get_embeddings_seq = reinterpret_cast<float*(*)(llama_context *, llama_seq_id)>(GetProcAddress(llama_dll, "llama_get_embeddings_seq"));
    model_has_encoder = reinterpret_cast<bool(*)(const llama_model *)>(GetProcAddress(llama_dll, "llama_model_has_encoder"));
    model_has_decoder = reinterpret_cast<bool(*)(const llama_model *)>(GetProcAddress(llama_dll, "llama_model_has_decoder"));
    model_n_ctx_train = reinterpret_cast<int32_t(*)(const llama_model *)>(GetProcAddress(llama_dll, "llama_model_n_ctx_train"));
    pooling_type = reinterpret_cast<enum llama_pooling_type(*)(const llama_context *)>(GetProcAddress(llama_dll, "llama_pooling_type"));
    //Sampler
    sampler_chain_default_params = reinterpret_cast<llama_sampler_chain_params(*)()>(GetProcAddress(llama_dll, "llama_sampler_chain_default_params"));
    sampler_chain_init = reinterpret_cast<llama_sampler*(*)(llama_sampler_chain_params)>(GetProcAddress(llama_dll, "llama_sampler_chain_init"));
    sampler_chain_add = reinterpret_cast<void(*)(llama_sampler *, llama_sampler *)>(GetProcAddress(llama_dll, "llama_sampler_chain_add"));
    sampler_init_grammar = reinterpret_cast<llama_sampler*(*)(const struct llama_vocab *, const char *, const char *)>(GetProcAddress(llama_dll, "llama_sampler_init_grammar"));
    sampler_reset = reinterpret_cast<void(*)(struct llama_sampler *)>(GetProcAddress(llama_dll,"llama_sampler_reset"));
    sampler_init_top_p = reinterpret_cast<llama_sampler*(*)(float, size_t)>(GetProcAddress(llama_dll, "llama_sampler_init_top_p"));
    sampler_init_min_p = reinterpret_cast<llama_sampler*(*)(float, size_t)>(GetProcAddress(llama_dll, "llama_sampler_init_min_p"));
    sampler_init_temp = reinterpret_cast<llama_sampler*(*)(float)>(GetProcAddress(llama_dll, "llama_sampler_init_temp"));
    sampler_init_dist = reinterpret_cast<llama_sampler*(*)(uint32_t)>(GetProcAddress(llama_dll, "llama_sampler_init_dist"));
    sampler_init_top_k = reinterpret_cast<llama_sampler*(*)(uint32_t)>(GetProcAddress(llama_dll, "llama_sampler_init_top_k"));
    //Memory
    memory_seq_rm = reinterpret_cast<bool(*)(llama_memory_t, llama_seq_id, llama_pos, llama_pos)>(GetProcAddress(llama_dll, "llama_memory_seq_rm"));
    memory_seq_pos_max = reinterpret_cast<llama_pos(*)(llama_memory_t, llama_seq_id)>(GetProcAddress(llama_dll, "llama_memory_seq_pos_max"));
    get_memory = reinterpret_cast<llama_memory_t(*)(const llama_context*)>(GetProcAddress(llama_dll, "llama_get_memory"));
    memory_clear = reinterpret_cast<void(*)(llama_memory_t, bool)>(GetProcAddress(llama_dll, "llama_memory_clear"));
    //Prompt
    tokenize = reinterpret_cast<int32_t(*)(const llama_vocab *, const char *, int32_t, llama_token *, int32_t, bool, bool)>(GetProcAddress(llama_dll, "llama_tokenize"));
    model_chat_template = reinterpret_cast<const char*(*)(const llama_model *, const char *)>(GetProcAddress(llama_dll, "llama_model_chat_template"));
    chat_apply_template = reinterpret_cast<int32_t(*)(const char *, const llama_chat_message *, size_t, bool, char *, int32_t)>(GetProcAddress(llama_dll, "llama_chat_apply_template"));
    batch_get_one = reinterpret_cast<llama_batch(*)(llama_token *, int32_t)>(GetProcAddress(llama_dll, "llama_batch_get_one"));
    batch_init = reinterpret_cast<llama_batch(*)(int32_t, int32_t, int32_t)>(GetProcAddress(llama_dll, "llama_batch_init"));
    decode = reinterpret_cast<int32_t(*)(llama_context *, llama_batch)>(GetProcAddress(llama_dll, "llama_decode"));
    sampler_sample = reinterpret_cast<llama_token(*)(llama_sampler *, llama_context *, int32_t)>(GetProcAddress(llama_dll, "llama_sampler_sample"));
    vocab_is_eog = reinterpret_cast<bool(*)(const llama_vocab *, llama_token)>(GetProcAddress(llama_dll, "llama_vocab_is_eog"));
    vocab_eos = reinterpret_cast<llama_token(*)(const llama_vocab*)>(GetProcAddress(llama_dll, "llama_vocab_eos"));
    vocab_bos = reinterpret_cast<llama_token(*)(const llama_vocab*)>(GetProcAddress(llama_dll, "llama_vocab_bos"));
    token_to_piece = reinterpret_cast<int32_t (*)(const llama_vocab *, llama_token, char *, int32_t, int32_t, bool)>(GetProcAddress(llama_dll, "llama_token_to_piece"));
    //Free
    model_free = reinterpret_cast<void(*)(llama_model*)>(GetProcAddress(llama_dll, "llama_model_free"));
    sampler_free = reinterpret_cast<void(*)(llama_sampler *)>(GetProcAddress(llama_dll, "llama_sampler_free"));
    llama_free = reinterpret_cast<void(*)(llama_context *)>(GetProcAddress(llama_dll, "llama_free"));
    batch_free = reinterpret_cast<void(*)(llama_batch)>(GetProcAddress(llama_dll, "llama_batch_free"));

    if (//Log
    !log_set ||
    !ggml_abort ||
    //Backend Load
    !ggml_backend_load_all ||
    !backend_init ||
    !backend_free ||
    //Model
    !context_default_params ||
    !model_default_params ||
    !model_load_from_file ||
    !model_get_vocab ||
    !init_from_model ||
    !n_ctx ||
    !model_n_embd ||
    !model_n_embd_out ||
    !get_embeddings ||
    !model_has_encoder ||
    !model_has_decoder ||
    !model_n_ctx_train ||
    !pooling_type ||
    //Sampler
    !sampler_chain_default_params ||
    !sampler_chain_init ||
    !sampler_chain_add ||
    !sampler_init_grammar ||
    !sampler_init_top_p ||
    !sampler_init_min_p ||
    !sampler_init_top_k ||
    !sampler_init_temp ||
    !sampler_init_dist ||
    //Memory
    !memory_seq_pos_max ||
    !get_memory ||
    !memory_clear ||
    //Prompt
    !tokenize ||
    !model_chat_template ||
    !chat_apply_template ||
    !batch_get_one ||
    !batch_init ||
    !decode ||
    !sampler_sample ||
    !vocab_is_eog ||
    !vocab_eos ||
    !vocab_bos ||
    !token_to_piece ||
    //Free
    !model_free ||
    !sampler_free ||
    !llama_free ||
    !batch_free)
    {
        last_error = "Failed to find required functions in Llama DLL.";
        cleanup();
        return false;
    }

    return true;
}
