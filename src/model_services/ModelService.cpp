#include "ModelService.h"

#include <utility>
#include "BackendService.h"

ModelService::ModelService(const ModelConfig & config, llama_model * model_ptr, llama_vocab * vocab_ptr) :
    config_(config),
    model_(model_ptr),
    vocab_(vocab_ptr) {
    auto & llama = BackendService::instance().llama();
    max_context_size_ = llama.model_n_ctx_train(model_);
    has_encoder_ = llama.model_has_encoder(model_);
    has_decoder_ = llama.model_has_decoder(model_);
}

ModelService::~ModelService() {
    if (model_ && BackendService::instance().is_running()) {
        BackendService::instance().llama().model_free(model_);
        if (config_.verbose) {
            fprintf(stdout, "[ModelService] Unloaded model.\n");
        }
    }
}



std::shared_ptr<ModelService> ModelService::create(const ModelConfig &config) {
    auto & backend = BackendService::instance();
    if (!backend.is_running()) {
        backend.start();
    }
    auto & llama = backend.llama();

    auto model_ptr = load_model(config);
    if (!model_ptr)
        return nullptr;
    auto vocab_ptr = load_vocab(config, model_ptr);
    if (!vocab_ptr)
        return nullptr;

    return std::make_shared<ModelService>(config, model_ptr, vocab_ptr);
}

int ModelService::n_embd() const {
    return BackendService::instance().llama().model_n_embd(model_);
}

int ModelService::n_embd_out() const {
    return BackendService::instance().llama().model_n_embd_out(model_);
}

llama_model * ModelService::load_model(const ModelConfig &config) {
    auto & llama = BackendService::instance().llama();
    llama_model_params model_params = llama.model_default_params();
    model_params.n_gpu_layers = config.gpu_layers;
    model_params.use_mmap = config.use_mmap;
    if (config.verbose) {
        fprintf(stdout, "[ModelService] Loading model - ");
        fflush(stdout);
    }
    llama_model * model_ptr = llama.model_load_from_file(config.model_path.c_str(), model_params);
    if (!model_ptr) {
        fprintf(stderr, "[ModelService] Failed to load model from: '%s'.\n", config.model_path.c_str());
        return nullptr;
    }

    if (config.verbose) {
        fprintf(stdout, "DONE\n");
        fflush(stdout);
    }
    return model_ptr;
}

llama_vocab * ModelService::load_vocab(const ModelConfig &config, const llama_model * model_ptr) {
    auto & llama = BackendService::instance().llama();
    if (config.verbose) {
        fprintf(stdout, "[ModelService] Loading vocab - ");
        fflush(stdout);
    }
    llama_vocab * vocab_ptr = llama.model_get_vocab(model_ptr);
    if (!vocab_ptr) {
        fprintf(stderr, "[ModelService] Failed to load model vocab.\n");
        return nullptr;
    }

    if (config.verbose) {
        fprintf(stdout, "DONE\n");
        fflush(stdout);
    }
    return vocab_ptr;
}

std::string ModelService::get_model_chat_template() const {
    auto & llama = BackendService::instance().llama();
    try {
        const char * template_string = llama.model_chat_template(model_, NULL);
        if (template_string) {
            return std::string(template_string);
        }

    }
    catch (...) {
        fprintf(stderr, "[ModelService] Failed to get chat template\n");
    }
    return std::string();
}
