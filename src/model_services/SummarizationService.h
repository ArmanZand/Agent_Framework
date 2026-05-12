#ifndef AGENT_FRAMEWORK_SUMMARIZATIONSERVICE_H
#define AGENT_FRAMEWORK_SUMMARIZATIONSERVICE_H
#include <string>

#include "PromptService.h"

struct SummarizationConfig {
    int private_context_size = 6096;
    int max_output_tokens = 6000;
    float temperature = 0.2f;
    std::string summarize_prompt_prefix =
        "Summarize the following conversation concisely (max 3 sentences).\n"
        "Do not add meta-commentary when there is little content.\n"
        "Look to include any key facts, decisions, or unique statements.\n"
        "Output only the summary, no preamble.\n";

    std::string extract_query_prefix =
        "Extract search keywords from this message using only words present in or directly named by the message. "
        "Do not add related concepts, synonyms, or background knowledge. "
        "Output only the keywords, nothing else.\n";

    std::string consolidate_prompt_prefix =
        "Extract up to 5 discrete facts or decisions from this exchange. "
        "Each fact must be single self-contained sentence under 20 words. ";
    std::string think_tag_open = "<think>";
    std::string think_tag_close = "</think>";
    bool verbose = false;
};

class SummarizationService {
public:
    SummarizationService(std::shared_ptr<PromptService> prompt_service,std::shared_ptr<ModelService> model_service, const SummarizationConfig & config = {});
    ~SummarizationService() = default;

    std::string generate_clean(const std::string & instruction);

    std::string summarize(const std::vector<ChatMessage>& messages);
    std::string extract_query(const std::string & message);
    std::vector<std::string> extract_facts(const std::string & user_message, const std::string & assistant_response);
    std::string consolidate_prompt(const std::string & user_message, const std::string & assistant_response);
    std::vector<llama_token> summarize_as_tokens(const std::vector<ChatMessage>& messages);

    const SummarizationConfig & config() const { return config_; }
private:
    std::string build_summarization_prompt(const std::vector<ChatMessage> & messages) const;
    std::string build_extract_query_prompt(const std::string & message) const;
    std::string build_consolidate_prompt(const std::string & user_message, const std::string & assistant_response);
    std::string trim(const std::string & input) const;
    std::string strip_thinking(std::string & input);
    std::vector<std::string> parse_string_list(const std::string &text) const;

    std::shared_ptr<PromptService> prompt_;
    std::shared_ptr<ModelService> model_;
    std::unique_ptr<SamplerService> sampler_;
    std::unique_ptr<ContextService> private_context_;
    SummarizationConfig config_;

};


#endif //AGENT_FRAMEWORK_SUMMARIZATIONSERVICE_H