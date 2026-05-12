#ifndef AGENT_FRAMEWORK_ITOOL_H
#define AGENT_FRAMEWORK_ITOOL_H

#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

enum ToolArgType {
    String,
    Number,
    Boolean,
    Array,
    Object
};

struct ToolParameter {
    std::string name;
    ToolArgType type;
    std::string description;
    bool required = false;
    json default_value = nullptr;

    json to_json() const {
        auto tooltype_to_string = [](const ToolArgType type) {
            switch (type) {
                default:
                case ToolArgType::String:
                    return "string";
                case ToolArgType::Number:
                    return "number";
                case ToolArgType::Boolean:
                    return "boolean";
                case ToolArgType::Array:
                    return "array";
                case ToolArgType::Object:
                    return "object";
            }
        };
        json j = {
            {"type", tooltype_to_string(type) },
            {"description", description}
        };
        if (!default_value.empty()) {
            j["default_value"] = default_value;
        }
        return j;
    }
};

struct ToolResult {
    bool success = false;
    std::string result;
    std::string error;
    std::map<std::string, std::string> metadata;

    json to_json() const {
        json j = {
            {"success", success},
            {"result", result}
        };
        if (!error.empty()) {
            j["error"] = error;
        }
        if (!metadata.empty()) {
            j["metadata"] = metadata;
        }
        return j;
    };
};

struct ToolDefinition {
    std::string name;
    std::string description;
    std::vector<ToolParameter> parameters;

    json to_json() const {
        json properties = json::object();
        json required_list  = json::array();

        for (const auto & param : parameters) {
            properties[param.name] = param.to_json();
            if (param.required) {
                required_list.push_back(param.name);
            }
        }
        return {
            {"name", name},
            {"description", description},
            {"parameters", {
            {"type", "object"},
                {"properties", properties},
                {"required", required_list}
            }}
        };
    };
};

class ITool {
public:
    virtual ~ITool() = default;
    virtual ToolDefinition  get_definition() const = 0;
    virtual ToolResult      execute(const json &arguments) = 0;
    std::string get_name() const { return get_definition().name; }
};

struct ToolCall {
    std::string id;
    std::string name;
    json arguments;

    json to_json() const {
        return {
        {"id", id},
        {"name", name},
        {"arguments", arguments}
        };
    }
};

#endif //AGENT_FRAMEWORK_ITOOL_H