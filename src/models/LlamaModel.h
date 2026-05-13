#ifndef AGENT_FRAMEWORK_LLAMAMODEL_H
#define AGENT_FRAMEWORK_LLAMAMODEL_H
#include "LlamaWrapper.h"

struct ModelConfig {
    std::string model_path;
    std::string model_name;
    int n_gpu_layers = 99;
    int n_ctx = 2048;
    int n_batch = 2048;
    int n_threads = 1;
    int n_threads_batch = 1;
    bool verbose = false;
};

class LlamaModel {
public:
    ~LlamaModel();

    static std::optional<LlamaModel> create(ModelConfig config_);

    // Get general info
    int get_context_size() const { return config.n_ctx; }
    std::string get_model_name() const { return config.model_name; };


    // Expose llama.cpp functions for agents
    LlamaWrapper & get_llama() { return llama; }
    llama_model * get_llama_model() { return model; }
    const llama_vocab * get_vocab() { return vocab; }

    const std::string & get_last_error() { return last_error; }

    // No copy
    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;
    // Can std::move
    LlamaModel(LlamaModel&&) noexcept;
    LlamaModel& operator=(LlamaModel&&) noexcept;

private:
    LlamaWrapper llama;
    llama_model * model;
    const llama_vocab * vocab;
    ModelConfig config;
    std::string last_error;

    LlamaModel(ModelConfig config_);
    bool initialize();
};


#endif //AGENT_FRAMEWORK_LLAMAMODEL_H