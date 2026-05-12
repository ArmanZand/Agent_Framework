#ifndef AGENT_FRAMEWORK_PROMPTSERVICE_H
#define AGENT_FRAMEWORK_PROMPTSERVICE_H
#include <functional>
#include <memory>
#include <string>
#include "ContextService.h"
#include "ModelService.h"
#include "SamplerService.h"
#include "../infrastructure/ChatMessage.h"


using TokenCallback = std::function<bool(const std::string & token_text)>;
using CompletionCallback = std::function<void(const std::string & full_response)>;


struct PromptConfig {
    int max_new_tokens = 0;
    bool add_bos = true;
    bool parse_special = true;
    bool echo_prompt = false;
    bool verbose = false;
    bool get_default_template = true;
};

enum class GenerationStop { Eos, MaxTokens, ContextFull, Cancelled };

struct GenerationResult {
    std::string text;
    GenerationStop stop = GenerationStop::Eos;
    int tokens = 0;
    double ms = 0.0;
};

class PromptService {
public:
    explicit PromptService(std::shared_ptr<ModelService> model_service, const PromptConfig & config = {});

    std::vector<llama_token> tokenize(const std::string & text, bool add_special  = false, bool parse_special = true) const;
    std::string detokenize(llama_token token, bool lstrip = false, bool special = false) const;
    std::string detokenize(const std::vector<llama_token> & tokens) const;

    int count_tokens(const std::string & text, bool add_special = false, bool parse_special = true) const;
    int count_chat_tokens(const std::vector<ChatMessage> & messages, bool add_assistant_turn) const;

    std::string render_chat(const std::vector<ChatMessage>& messages, bool add_assistant_turn = true) const;
    std::string render_turn(const std::string & role, const std::string & content, bool add_assistant_turn = true) const;
    std::string render_partial_assistant(const std::string & content) const;
    int decode_batch(ContextService & ctx, const std::vector<llama_token>& tokens) const;

    std::string generate(
        ContextService & ctx,
        SamplerService& sampler,
        const std::string & prompt,
        TokenCallback on_token = nullptr,
        CompletionCallback on_done = nullptr) const;


    std::string generate_chat(
        ContextService & ctx,
        SamplerService & sampler,
        const std::vector<ChatMessage>& messages,
        TokenCallback on_token = nullptr,
        CompletionCallback on_done = nullptr) const;

    GenerationResult generate_continue(ContextService &ctx, SamplerService &sampler, TokenCallback on_token);

    std::vector<float> generate_embedding(ContextService & ctx, const std::string & text) const;
    std::vector<std::vector<float>> generate_embeddings(ContextService & ctx, const std::vector<std::string> & texts) const;
    const PromptConfig & config() const { return config_; }

private:
    std::string bos_token_str_;
    std::string eos_token_str_;
    std::string chat_template_;

    static std::string token_to_string(const llama_vocab* vocab, llama_token  token);
    void init_template(const llama_vocab* vocab);
    std::shared_ptr<ModelService> model_;
    PromptConfig config_;
};


#endif //AGENT_FRAMEWORK_PROMPTSERVICE_H