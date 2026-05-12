#include "ToolRegistry.h"
#include <sstream>

void ToolRegistry::register_tool(std::shared_ptr<ITool> tool) {
    if (tool) {
        tools[tool->get_name()] = tool;
    }
}

std::shared_ptr<ITool> ToolRegistry::get_tool(const std::string & name) const {
    auto it = tools.find(name);
    return (it != tools.end()) ? it->second : nullptr;
}

std::vector<ToolDefinition> ToolRegistry::get_all_definitions() const {
    std::vector<ToolDefinition> result;
    for (const auto & [name, tool] : tools) {
        result.push_back(tool->get_definition());
    }
    return result;
}

ToolResult ToolRegistry::execute_tool(const std::string &name, const json & arguments) const {
    auto tool = get_tool(name);
    if (!tool) {
        ToolResult error;
        error.error = "Tool not found: " + name;
        return error;
    }

    return tool->execute(arguments);
}

bool ToolRegistry::has_tool(const std::string &name) const {
    return tools.find(name) != tools.end();
}

void ToolRegistry::remove_tool(const std::string &name) {
    tools.erase(name);
}

std::vector<std::string> ToolRegistry::get_tool_names() const {
    std::vector<std::string> result;
    for (const auto & [name, tool] : tools) {
        result.push_back(name);
    }
    return result;
}

std::string ToolRegistry::format_tools_for_prompt() const {
    json tools_json = json::array();

    for (const auto & [name, tool] : tools) {
        tools_json.push_back(tool->get_definition().to_json());
    }
    return tools_json.dump(2);
}
