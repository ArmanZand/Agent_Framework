#ifndef AGENT_FRAMEWORK_LLAMAMODEL_H
#define AGENT_FRAMEWORK_LLAMAMODEL_H
#include "../interfaces/IModel.h"
#include "LlamaWrapper.h"

class LlamaModel  : public IModel {
public:
    ~LlamaModel();

    static std::optional<LlamaModel> create(ModelConfig config_);

    // IModel implementations
    GenerateResult generate(const GenerateParams & params) override;
    std::unique_ptr<Session> create_session(int n_ctx = 2048) override;
    GenerateResult generate_with_session(Session* session, const GenerateParams & params) override;

    // Get general info
    int get_context_size() const override { return config.n_ctx; }
    std::string get_model_name() const override { return config.model_name; };

    // Llama Session
    class LlamaSession : public Session {
    public:
        llama_context * ctx;
        llama_sampler * sampler;
        LlamaWrapper * llama;

        LlamaSession(llama_context * c, llama_sampler * s, LlamaWrapper * l) : ctx(c), sampler(s), llama(l) {}
        ~LlamaSession() {
            if (sampler) llama->sampler_free(sampler);
            if (ctx) llama->llama_free(ctx);
        }
    };

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