#include "EmbeddingModel.h"

#include <memory>

#include "LlamaModel.h"
#include "LlamaWrapper.h"
#include <cmath>

void EmbeddingModel::normalize_embedding(std::vector<float> &embedding) {
    const int n = embedding.size();
    double sum = 0.0;

    switch (norm_type) {
        case NormalizationType::NONE:
            return;
        case NormalizationType::MAX_ABS:
            // max absolute value normalization (int16 range)
            for (int i = 0; i < n;i++) {
                if (sum < std::abs(embedding[i])) {
                    sum = std::abs(embedding[i]);
                }
            }
            sum /= 32760.0; //scale to int16 range
            break;
        case NormalizationType::EUCLIDEAN:
            // standard for embeddings
            for (int i = 0; i < n; i++) {
                sum += embedding[i] * embedding[i];
            }
            sum = std::sqrt(sum);
            break;
        case NormalizationType::P_NORM:
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(embedding[i]), p_norm_value);
            }
            sum = std::pow(sum, 1.0 / p_norm_value);
            break;
    }
    const float norm = sum > 0.0 ? 1.0f / sum : 0.0f;
    for (int i = 0;i < n; i++) {
        embedding[i] *= norm;
    }
}

EmbeddingModel::EmbeddingModel(const Config &cfg)
    : model_path(cfg.model_path),
    dimension(0),
    is_loaded(false),
    n_ctx(cfg.n_ctx),
    n_gpu_layers(cfg.n_gpu_layers),
    verbose(cfg.verbose),
    norm_type(cfg.normalization),
    p_norm_value(cfg.p_norm_value),
    pooling_type(cfg.pooling)


{
    state.llama_wrapper = nullptr;
    state.model = nullptr;
    state.vocab = nullptr;
}

EmbeddingModel::~EmbeddingModel() {
    unload();
}

bool EmbeddingModel::load() {
    if (is_loaded) return true;

    if (verbose) {
        printf("Loading embedding model: %s\n", model_path.c_str());
        fflush(stdout);
    }

    auto llama_opt = LlamaWrapper::initialize();
    if (!llama_opt) {
        fprintf(stderr, "Failed to initialize LlamaWrapper\n");
        return false;
    }

    auto llama = std::make_unique<LlamaWrapper>(std::move(*llama_opt));

    static void (*log_callback)(enum ggml_log_level, const char*, void*) =
        [](enum ggml_log_level level, const char * text, void*) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
               fprintf(stderr, "%s", text);
            }
    };
    llama->log_set(log_callback, nullptr);

    llama->ggml_backend_load_all();
    llama->backend_init();

    llama_model_params model_params = llama->model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama->model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load embedding model\n");
        return false;
    }

    if (llama->model_has_encoder(model) && llama->model_has_decoder(model)) {
        fprintf(stderr, "Error: Encoder-decoder models are not supported for embeddings\n");
        llama->model_free(model);
        return false;
    }

    int n_ctx_train_val = llama->model_n_ctx_train(model);
    if (n_ctx > n_ctx_train_val) {
        printf("Warning: model has trained only %d context tokens (%d specified)\n", n_ctx_train_val, n_ctx);
    }

    const llama_vocab * vocab = llama->model_get_vocab(model);
    if (!vocab) {
        fprintf(stderr, "Failed to get vocabulary\n");
        llama->model_free(model);
        return false;
    }

    dimension = llama->model_n_embd_out(model);


    state.llama_wrapper = llama.release();
    state.model = model;
    state.vocab = (void*)vocab;
    is_loaded = true;

    if (verbose) {
        printf("✓ Embedding model loaded\n");
        printf("  Dimension: %d\n", dimension);
        fflush(stdout);
    }

    return true;
}

void EmbeddingModel::unload() {
    if (!is_loaded)
        return;

    if (verbose) {
        printf("Unloading embedding model...\n");
        fflush(stdout);
    }

    auto * llama = static_cast<LlamaWrapper *>(state.llama_wrapper);
    auto * model = static_cast<llama_model *>(state.model);

    if (model && llama) {
        llama->model_free(model);
    }
    delete llama;

    state.llama_wrapper = nullptr;
    state.model = nullptr;
    state.vocab = nullptr;

    is_loaded = false;

    if (verbose) {
        printf("✓ Embedding model unloaded\n");
        fflush(stdout);
    }
}

std::vector<float> EmbeddingModel::embed(const std::string &text) {
    if (!is_loaded) {
        fprintf(stderr, "Embedding model not loaded\n");
        return std::vector<float>();
    }

    auto * llama = static_cast<LlamaWrapper *>(state.llama_wrapper);
    auto * model = static_cast<llama_model *>(state.model);
    auto * vocab = static_cast<const llama_vocab *>(state.vocab);

    if (!llama || !model || !vocab) {
        fprintf(stderr, "Invalid embedding model state\n");
        return std::vector<float>();
    }

    llama_context_params ctx_params = llama->context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;
    ctx_params.embeddings = true;
    switch (pooling_type) {
        case PoolingType::NONE:
            ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
            break;
        case PoolingType::MEAN:
            ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
            break;
        case PoolingType::CLS:
            ctx_params.pooling_type = LLAMA_POOLING_TYPE_CLS;
            break;
        case PoolingType::LAST:
            ctx_params.pooling_type = LLAMA_POOLING_TYPE_LAST;
            break;
    }
    //TODO: handle other pooling types
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;



    llama_context * ctx = llama->init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return std::vector<float>(dimension, 0.0f);
    }

    const int n_tokens = -llama->tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    if (n_tokens < 0) {
        llama->llama_free(ctx);
        fprintf(stderr, "Failed to tokenize text\n");
        return std::vector<float>(dimension, 0.0f);
    }

    std::vector<llama_token> tokens(n_tokens);
    llama->tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);

    if (tokens.size() > static_cast<size_t>(n_ctx)) {
        tokens.resize(n_ctx);
    }

    llama->memory_clear(llama->get_memory(ctx), true);

    llama_batch batch = llama->batch_init(tokens.size(), 0, 1);

    batch.n_tokens = 0;
    for (size_t i = 0; i < tokens.size(); i++) {
        // Check for overflow
        if (batch.n_tokens >= n_ctx) {
            fprintf(stderr, "Error: Batch overflow\n");
            llama->batch_free(batch);
            llama->llama_free(ctx);
            return std::vector<float>(dimension, 0.0f);
        }

        batch.token[batch.n_tokens] = tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = false;

        batch.n_tokens++;
    }
    if (batch.n_tokens > 0) {
        batch.logits[batch.n_tokens - 1] = true;
    }

    if (verbose) {
        printf("  Batch prepared: %d tokens\n", batch.n_tokens);
        printf("  First token: %d, Last token: %d\n",
               batch.token[0], batch.token[batch.n_tokens - 1]);
        fflush(stdout);
    }

    if (llama->decode(ctx, batch) != 0) {
        llama->llama_free(ctx);
        llama->batch_free(batch);
        fprintf(stderr, "Failed to decode batch\n");
        return std::vector<float>(dimension, 0.0f);
    }
    const enum llama_pooling_type pooling_type = llama->pooling_type(ctx);

    float * embeddings_raw = nullptr;
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        embeddings_raw = llama->get_embeddings_ith(ctx, batch.n_tokens - 1);
    }
    else {
        embeddings_raw = llama->get_embeddings_seq(ctx, 0);
    }

    if (!embeddings_raw) {
        llama->llama_free(ctx);
        llama->batch_free(batch);
        fprintf(stderr, "Failed to get embeddings\n");
        return std::vector<float>(dimension, 0.0f);
    }

    std::vector<float> embedding(embeddings_raw, embeddings_raw + dimension);


    normalize_embedding(embedding);

    llama->batch_free(batch);
    llama->llama_free(ctx);
    return embedding;

}
