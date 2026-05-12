#include "LlamaModel.h"
#include <iostream>

LlamaModel::LlamaModel(ModelConfig config_)
    : config(std::move(config_)), model(nullptr), vocab(nullptr) {}

LlamaModel::~LlamaModel() {
    if (model) {
        llama.model_free(model);
    }
}

LlamaModel::LlamaModel(LlamaModel && other ) noexcept
    : llama(std::move(other.llama)), model(other.model), vocab(other.vocab), config(std::move(other.config))  {
    other.model = nullptr;
    other.vocab = nullptr;
}

LlamaModel & LlamaModel::operator=(LlamaModel&& other) noexcept {
    if (this != &other) {
        if (model) {
            llama.model_free(model);
        }

        llama = std::move(other.llama);
        model = other.model;
        vocab = other.vocab;
        config = std::move(other.config);

        other.model = nullptr;
        other.vocab = nullptr;
    }
    return *this;
}

std::optional<LlamaModel> LlamaModel::create(ModelConfig config_) {
    LlamaModel llama_model(std::move(config_));

    if (!llama_model.initialize()) {
        return std::nullopt;
    }

    return llama_model;
}

bool LlamaModel::initialize() {
    // Initialize Llama Wrapper
    auto llama_opt = LlamaWrapper::initialize();
    if (!llama_opt) {
        last_error = "Failed to initialize Llama wrapper";
        return false;
    }
    llama = std::move(*llama_opt);

    // Set up logging
    llama.log_set([](enum ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    // Load backends
    llama.ggml_backend_load_all();
    llama.backend_init();

    // Load model
    llama_model_params model_params = llama.model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;

    if (config.verbose) {
        printf("Loading model from: %s\n", config.model_path.c_str());
    }

    model = llama.model_load_from_file(config.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model from: %s\n", config.model_path.c_str());
        return false;
    }

    vocab = llama.model_get_vocab(model);
    if (!vocab) {
        fprintf(stderr, "Failed to get vocabulary from model\n");
        return false;
    }

    if (config.verbose) {
        printf("Model loaded successfully\n");
    }

    return true;
}

IModel::GenerateResult LlamaModel::generate(const GenerateParams& params) {
    GenerateResult result;

    // Context
    llama_context_params ctx_params = llama.context_default_params();
    ctx_params.n_ctx = config.n_ctx;
    ctx_params.n_batch = config.n_batch;
    ctx_params.n_threads = config.n_threads;
    ctx_params.n_threads_batch = config.n_threads_batch;

    llama_context * ctx = llama.init_from_model(model, ctx_params);
    if (!ctx) {
        result.error = "Failed to create context";
        return result;
    }

    // Sampler
    llama_sampler* sampler = llama.sampler_chain_init(llama.sampler_chain_default_params());
    if (!sampler) {
        llama.llama_free(ctx);
        result.error = "Failed to create sampler";
        return result;
    }
    if (params.top_k > 0) {
        llama.sampler_chain_add(sampler, llama.sampler_init_top_k(params.top_k));
    }
    if (params.top_p < 1.0f) {
        llama.sampler_chain_add(sampler, llama.sampler_init_top_p(params.top_p, 1));
    }

    if (params.min_p > 0.0f) {
        llama.sampler_chain_add(sampler, llama.sampler_init_min_p(params.min_p, 1));
    }
    llama.sampler_chain_add(sampler, llama.sampler_init_temp(params.temperature));
    llama.sampler_chain_add(sampler, llama.sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Tokenize
    const bool is_first = llama.memory_seq_pos_max(llama.get_memory(ctx), 0) == -1;
    const int n_prompt_tokens = -llama.tokenize(vocab, params.prompt.c_str(), params.prompt.size(), nullptr, 0, is_first, true);

    if (n_prompt_tokens <= 0) {
        llama.sampler_free(sampler);
        llama.llama_free(ctx);
        result.error = "Failed to tokenize prompt";
        return result;
    }

    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama.tokenize(vocab, params.prompt.c_str(), params.prompt.size(),
                       prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        llama.sampler_free(sampler);
        llama.llama_free(ctx);
        result.error = "Failed to tokenize prompt";
        return result;
                       }

    // Generate
    llama_batch batch = llama.batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    int token_count = 0;
    while (true) {
        // Check context size
        int n_ctx_val = llama.n_ctx(ctx);
        int n_ctx_used = llama.memory_seq_pos_max(llama.get_memory(ctx), 0) + 1;
        if (n_ctx_used + batch.n_tokens > n_ctx_val) {
            break;
        }

        // Decode
        if (llama.decode(ctx, batch) != 0) {
            llama.sampler_free(sampler);
            llama.llama_free(ctx);
            result.error = "Decode failed";
            return result;
        }

        // Sample
        llama_token new_token_id = llama.sampler_sample(sampler, ctx, -1);

        // Check for end
        if (llama.vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        // Check max tokens
        if (params.max_tokens > 0 && token_count >= params.max_tokens) {
            break;
        }

        // Convert token to text
        char buf[256];
        int n = llama.token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            llama.sampler_free(sampler);
            llama.llama_free(ctx);
            result.error = "Token conversion failed";
            return result;
        }

        std::string token_str(buf, n);
        result.text += token_str;
        token_count++;

        // Call streaming callback if provided
        if (params.on_token) {
            params.on_token(token_str);
        }

        batch = llama.batch_get_one(&new_token_id, 1);
    }

    // Cleanup
    llama.sampler_free(sampler);
    llama.llama_free(ctx);

    result.tokens_generated = token_count;
    result.success = true;

    return result;
}

std::unique_ptr<IModel::Session> LlamaModel::create_session(int n_ctx) {
    llama_context_params ctx_params = llama.context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;

    llama_context * ctx = llama.init_from_model(model, ctx_params);
    if (!ctx) {
        last_error = "Failed to create session context";
        return nullptr;
    }

    llama_sampler * sampler = llama.sampler_chain_init(llama.sampler_chain_default_params());
    if (!sampler) {
        llama.llama_free(ctx);
        last_error = "Failed to create session sampler";
        return nullptr;
    }

    llama.sampler_chain_add(sampler, llama.sampler_init_dist(LLAMA_DEFAULT_SEED));

    return std::make_unique<LlamaSession>(ctx, sampler, &llama);
}

IModel::GenerateResult LlamaModel::generate_with_session(Session *session, const GenerateParams &params) {
    GenerateResult result;

    auto * llama_session = dynamic_cast<LlamaSession *>(session);
    if (!llama_session) {
        result.error = "Failed to create type";
        return result;
    }

    llama_context * ctx = llama_session->ctx;
    if (llama_session->sampler) {
        llama.sampler_free(llama_session->sampler);
    }

    llama_sampler * sampler = llama.sampler_chain_init(llama.sampler_chain_default_params());
    if (!sampler) {
        result.error = "Failed to create sampler for session";
        return result;
    }
    if (params.top_k > 0) {
        llama.sampler_chain_add(sampler, llama.sampler_init_top_k(params.top_k));
    }
    if (params.top_p < 1.0f) {
        llama.sampler_chain_add(sampler, llama.sampler_init_top_p(params.top_p, 1));
    }

    if (params.min_p > 0.0f) {
        llama.sampler_chain_add(sampler, llama.sampler_init_min_p(params.min_p, 1));
    }
    llama.sampler_chain_add(sampler, llama.sampler_init_temp(params.temperature));
    llama.sampler_chain_add(sampler, llama.sampler_init_dist(LLAMA_DEFAULT_SEED));

    llama_session->sampler = sampler;


    const bool is_first = llama.memory_seq_pos_max(llama.get_memory(ctx), 0) == -1;
    const int n_prompt_tokens = -llama.tokenize(vocab, params.prompt.c_str(), params.prompt.size(), nullptr, 0,
        is_first, true);
    if (n_prompt_tokens <= 0) {
        result.error = "Failed to tokenize prompt";
        return result;
    }

    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama.tokenize(vocab, params.prompt.c_str(), params.prompt.size(), prompt_tokens.data(), prompt_tokens.size(),
        is_first, true) < 0) {
        result.error = "Failed to tokenize prompt";
        return result;
    }

    llama_batch batch = llama.batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    int token_count = 0;

    while (true) {
        int n_ctx_val = llama.n_ctx(ctx);
        int n_ctx_used = llama.memory_seq_pos_max(llama.get_memory(ctx), 0) + 1;

        if (n_ctx_used + batch.n_tokens > n_ctx_val) {
            result.error = "Context size exceeded";
            break;
        }

        if (llama.decode(ctx, batch) != 0) {
            result.error = "Decode failed";
            return result;
        }

        llama_token new_token_id = llama.sampler_sample(sampler, ctx, -1);

        if (llama.vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        if (params.max_tokens > 0 && token_count >= params.max_tokens) {
            break;
        }

        char buf[256];
        int n = llama.token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            result.error = "Token conversion failed";
            return result;
        }

        std::string token_str(buf, n);
        result.text += token_str;
        token_count++;

        if (params.on_token) {
            params.on_token(token_str);
        }

        batch = llama.batch_get_one(&new_token_id, 1);
    }

    result.tokens_generated = token_count;
    result.success = true;
    return result;
}
