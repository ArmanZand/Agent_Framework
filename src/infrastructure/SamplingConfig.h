#ifndef AGENT_FRAMEWORK_SAMPLINGCONFIG_H
#define AGENT_FRAMEWORK_SAMPLINGCONFIG_H
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

struct SamplingConfig {
    float temperature = 0.7f;
    float top_p = 0.8f;
    float min_p = 0.0f;
    int top_k = 20;
    int max_tokens = -1;

    json to_json() const {
        return {
                {"temperature", temperature},
                {"top_p", top_p},
                {"min_p", min_p},
                {"top_k", top_k},
                {"max_tokens", max_tokens}
        };
    }

    static SamplingConfig from_json(const json & j) {
        SamplingConfig cfg;
        if (j.contains("temperature"))  cfg.temperature = j["temperature"];
        if (j.contains("top_p"))        cfg.top_p = j["top_p"];
        if (j.contains("min_p"))        cfg.min_p = j["min_p"];
        if (j.contains("top_k"))        cfg.top_k = j["top_k"];
        if (j.contains("max_tokens"))   cfg.max_tokens = j["max_tokens"];
        return cfg;
    }
};
#endif //AGENT_FRAMEWORK_SAMPLINGCONFIG_H