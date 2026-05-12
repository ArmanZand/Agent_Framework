#include "SummarizationService.h"

#include <algorithm>
#include <sstream>
#include "nlohmann/json.hpp"
using json = nlohmann::ordered_json;

SummarizationService::SummarizationService(std::shared_ptr<PromptService> prompt_service,
                                           std::shared_ptr<ModelService> model_service, const SummarizationConfig & config)
        :   prompt_(std::move(prompt_service)),
            model_(std::move(model_service)),
            config_(config) {
    ContextConfig context_config;
    context_config.context_size = config_.private_context_size;
    private_context_ = std::make_unique<ContextService>(model_, context_config);
    private_context_->initialize();

    SamplerConfig sc;
    sc.top_k = 1;
    sc.top_p = 1.0f;
    sc.min_p = 0.0f;
    sc.temperature = config_.temperature;
    sampler_ = std::make_unique<SamplerService>(sc);
    sampler_->initialize();
}

std::string SummarizationService::generate_clean(const std::string &instruction) {
    std::string formatted = prompt_->render_turn("user", instruction, true);
    private_context_->clear_kv_cache();
    std::string raw = prompt_->generate(*private_context_, *sampler_, formatted);
    private_context_->clear_kv_cache();
    return trim(strip_thinking(raw));
}

std::string SummarizationService::summarize(const std::vector<ChatMessage> &messages) {
    if (messages.empty()) return {};
    if (config_.verbose)
        fprintf(stdout, "[SummarizationService] Summarizing %zu messages...\n", messages.size());
    std::string result = generate_clean(build_summarization_prompt(messages));
    if (config_.verbose)
        fprintf(stdout, "[SummarizationService] Summary: %s\n", result.c_str());
    return result;
}

std::string SummarizationService::extract_query(const std::string &message) {
    if (message.empty()) return {};
    return generate_clean(build_extract_query_prompt(message));
}

std::vector<std::string> SummarizationService::extract_facts(const std::string & user_message,
    const std::string &assistant_response)  {
    std::string prompt =
        "Extract self-contained facts from this exchange. "
        "Each string must be a single present-tense sentence with a specific value, "
        "file path, name, figure, structure, decision, or domain fact worth recalling. "
        "Opinions, recommendations, and assessments must include their source and rationale "
        "to be worth storing — skip them if that context is absent. "
        "When extracting figures, write abbreviated form where natural (£842.4m, not £842,400,000) and remove comma delimiters from unabbreviated numbers."
        "Skip generic observations. Output [] if there are no meaningful facts. "
        "Output a JSON array of strings — no preamble, no markdown fences."
        "Example: [\"Orion Dynamics revenue was £842.4m in Q3 2024.\", \"Quarterly results file is at finance/company_reports/orion_dynamics_q3_2024.txt\"] "
        "Always name the subject explicitly in each fact — never use pronouns or implicit references. "
        "Include a time reference (date, quarter, or year) wherever one is available in the source."
        "\n\n"
        "User: " + user_message + "\n\nAssistant: "  + assistant_response;
    if (config_.verbose) {
        fprintf(stdout, "[SummarizationService] Extracting key words for querying...");
    }
    return parse_string_list(generate_clean(prompt));
}

std::string SummarizationService::consolidate_prompt(const std::string &user_message,
                                                     const std::string &assistant_response) {
    if (user_message.empty()) return {};
    return generate_clean(build_consolidate_prompt(user_message, assistant_response));
}

std::vector<llama_token> SummarizationService::summarize_as_tokens(const std::vector<ChatMessage> &messages) {
    std::string summary = summarize(messages);
    if (summary.empty()) return {};
    return prompt_->tokenize(summary, false, true);
}

std::string SummarizationService::build_summarization_prompt(const std::vector<ChatMessage> &messages) const {
    std::ostringstream oss;
    oss << config_.summarize_prompt_prefix << "\n";
    for (const auto & m : messages) {
        oss << m.role << ": " << m.content << "\n";
    }
    oss << "\nSummary:";
    return oss.str();
}

std::string SummarizationService::build_extract_query_prompt(const std::string &message) const {
    std::stringstream oss;
    oss << config_.extract_query_prefix << "\n";
    oss << "Message: " + message;
    return oss.str();
}

std::string SummarizationService::build_consolidate_prompt(const std::string &user_message,
    const std::string &assistant_response) {
    std::stringstream oss;
    oss << config_.consolidate_prompt_prefix << "\n";
    oss << "User: " << user_message << + "\n";
    oss << "Assistant: " << assistant_response + "\n\n";
    oss << "Facts:";
    return oss.str();
}

std::string SummarizationService::trim(const std::string &input) const {
    size_t str_begin = input.find_first_not_of(" \t");
    if (str_begin == std::string::npos)
        return input;

    size_t str_end = input.find_last_not_of(" \t");
    return input.substr(str_begin, str_end - str_begin + 1);
}

std::string SummarizationService::strip_thinking(std::string & input) {
    std::string text = trim(input);
    for (;;) {
        size_t ts = text.find(config_.think_tag_open);
        if (ts == std::string::npos) break;
        size_t te = text.find(config_.think_tag_close, ts);
        if (te == std::string::npos) { text.erase(ts);  break; }
        text.erase(ts, te + config_.think_tag_close.size() - ts);
    }
    return text;
}

std::vector<std::string> SummarizationService::parse_string_list(const std::string &text) const {
    std::vector<std::string> results;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t open = text.find('"', pos);
        if (open == std::string::npos) break;
        size_t close = open + 1;
        while (close < text.size()) {
            if (text[close] == '\\') { close += 2; continue; }
            if (text[close] == '"')  break;
            ++close;
        }
        if (close >= text.size()) break;

        std::string s;
        s.reserve(close - open);
        for (size_t i = open + 1; i < close; ++i) {
            if (text[i] == '\\' && i + 1 < close) {
                ++i;
                switch (text[i]) {
                    case '"':  s += '"';  break;
                    case '\\': s += '\\'; break;
                    case 'n':  s += '\n'; break;
                    case 't':  s += '\t'; break;
                    default:   s += text[i]; break;
                }
            } else {
                s += text[i];
            }
        }
        pos = close + 1;
        if (s.size() >= 10) {
            results.push_back(std::move(s));
        }
    }
    if (!results.empty()) return results;

    // Fallback
    static constexpr const char* skip_prefixes[] = {
        "Let me", "I'll", "I need", "I have", "Looking at", "Going through",
        "Now I",  "Here are", "Based on", "Below are", "These are", "I will"
    };
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.size() >= 2 && line[1] == ':') continue;
        if (line.rfind("Assistant:", 0) == 0) continue;
        if (!line.empty() && line.back() == ':') continue;
        if (line.size() >= 3 && line.substr(line.size() - 3) == ":**") continue;
        bool skip = false;
        for (const auto* p : skip_prefixes)
            if (line.rfind(p, 0) == 0) { skip = true; break; }
        if (skip) continue;
        size_t start = line.find_first_not_of(" \t-*•0123456789.)\"");
        if (start == std::string::npos) continue;
        std::string fact = line.substr(start);
        while (!fact.empty() && (fact.back() == '"' || fact.back() == ',' || fact.back() == ' '))
            fact.pop_back();
        if (fact.size() < 10) continue;
        results.push_back(std::move(fact));
    }
    return results;
}
