#ifndef AGENT_FRAMEWORK_MODELSERVICE_H
#define AGENT_FRAMEWORK_MODELSERVICE_H
#include <memory>
#include <string>
#include "../models/LlamaWrapper.h"

struct ModelConfig {
    std::string model_path;
    int gpu_layers = 99;
    bool use_mmap = true;
    bool verbose = false;
};
class ModelService {
public:
    ModelService(const ModelConfig & config, llama_model * model_ptr, llama_vocab * vocab_ptr);
    ~ModelService();

    ModelService(const ModelService &) = delete;
    ModelService & operator=(const ModelService &) = delete;

    static std::shared_ptr<ModelService> create(const ModelConfig & config);

    llama_model * model()   const { return model_; }
    llama_vocab * vocab()   const { return vocab_; }
    bool has_encoder()      const { return has_encoder_; }
    bool has_decoder()      const { return has_decoder_; }
    int max_context_size()  const { return max_context_size_; }

    std::string get_model_chat_template() const;

    int n_embd() const;
    int n_embd_out() const;
private:
    static llama_model * load_model(const ModelConfig &config);
    static llama_vocab *  load_vocab(const ModelConfig &config, const llama_model * model_ptr);

    ModelConfig config_;
    llama_vocab * vocab_ = nullptr;
    llama_model * model_ = nullptr;;
    bool has_encoder_ = false;
    bool has_decoder_ = false;
    int max_context_size_ = 0;
};


#endif //AGENT_FRAMEWORK_MODELSERVICE_H