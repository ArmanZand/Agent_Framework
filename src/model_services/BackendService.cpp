#include "BackendService.h"

BackendService::BackendService() {
}

BackendService::~BackendService() {
    stop();
}

bool BackendService::start() {
    if (is_running_) return true;
    auto llama_opt = LlamaWrapper::initialize();
    if (!llama_opt) {
        fprintf(stderr, "[BackendService] Failed to initialize Llama wrapper.\n");
        return false;
    }
    llama_ = std::move(*llama_opt);

    llama_.ggml_backend_load_all();
    llama_.backend_init();


    llama_.log_set([](enum ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
    }, nullptr);

    is_running_ = true;
    fprintf(stdout, "[BackendService] Started.\n");
    return true;
}

void BackendService::stop() {
    if (is_running_) {
        llama_.backend_free();
        is_running_ = false;
        fprintf(stdout, "[BackendService] Stopped.\n");
    }
}


