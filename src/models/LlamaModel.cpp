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
