#ifndef AGENT_FRAMEWORK_MEMORYCONFIG_H
#define AGENT_FRAMEWORK_MEMORYCONFIG_H
#include <functional>
#include <string>
#include "../memory/Summarizer.h"
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

enum class CompressionStrategy {
    None,
    Summarize,
    DropOldest,
    SlidingWindow,
    ImportanceWeighted
};

NLOHMANN_JSON_SERIALIZE_ENUM(CompressionStrategy, {
    {CompressionStrategy::None, "None"},
    {CompressionStrategy::Summarize, "Summarize"},
    {CompressionStrategy::DropOldest, "DropOldest"},
    {CompressionStrategy::SlidingWindow, "SlidingWindow"},
    {CompressionStrategy::ImportanceWeighted, "ImportanceWeighted"}
})

struct MemoryConfig {
    int max_context_tokens = 4096;
    int reserved_system_tokens = 512;
    int reserved_tools_tokens = 256;
    int min_keep_messages = 5;
    CompressionStrategy compression_strategy = CompressionStrategy::SlidingWindow;
    int sliding_window_size = 20;
    int summary_target_tokens = 256;
    std::function<std::string(const std::string&, IModel &, int)> summarizer;
    bool use_exact_token_count = true;
    float token_multiplier = 4.0f;
    bool verbose = false;

    json to_json() const {
        return {
            {"max_context_tokens", max_context_tokens},
            {"reserved_system_tokens", reserved_system_tokens},
            {"reserved_tools_tokens", reserved_tools_tokens},
            {"min_keep_messages", min_keep_messages},
            {"compression_strategy", compression_strategy},
            {"sliding_window_size", sliding_window_size},
            {"summary_target_tokens", summary_target_tokens},
            {"use_exact_token_count", use_exact_token_count},
            {"token_multiplier", token_multiplier},
            {"verbose", verbose}
        };
    }

    static MemoryConfig from_json(const json & j) {
        MemoryConfig cfg;
        cfg.max_context_tokens = j.value("max_context_tokens", 4096);
        cfg.reserved_system_tokens = j.value("reserved_system_tokens", 512);
        cfg.reserved_tools_tokens = j.value("reserved_tools_tokens", 256);
        cfg.min_keep_messages = j.value("min_keep_messages", 5);
        cfg.compression_strategy = j.value("compression_strategy", CompressionStrategy::SlidingWindow);
        cfg.sliding_window_size = j.value("sliding_window_size", 20);
        cfg.summary_target_tokens = j.value("summary_target_tokens", 256);
        cfg.summarizer = Summarizer::summarize;
        cfg.use_exact_token_count = j.value("use_exact_token_count", true);
        cfg.token_multiplier = j.value("token_multiplier", 4.0f);
        cfg.verbose = j.value("verbose", false);
        return cfg;

    }
};

#endif //AGENT_FRAMEWORK_MEMORYCONFIG_H