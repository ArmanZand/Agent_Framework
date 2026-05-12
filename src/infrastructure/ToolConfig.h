#ifndef AGENT_FRAMEWORK_TOOLCONFIG_H
#define AGENT_FRAMEWORK_TOOLCONFIG_H
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

struct ToolConfig {
    bool enabled = true;
    int max_iterations = 9;

    json to_json() const {
        return {
                {"enabled", enabled},
                {"max_iterations", max_iterations}
        };
    }

    static ToolConfig from_json(const json & j) {
        ToolConfig cfg;
        cfg.enabled = j.value("enabled", false);
        cfg.max_iterations = j.value("max_iterations", 5);
        return cfg;
    }
};
#endif //AGENT_FRAMEWORK_TOOLCONFIG_H