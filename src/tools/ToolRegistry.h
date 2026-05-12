#ifndef AGENT_FRAMEWORK_TOOLREGISTRY_H
#define AGENT_FRAMEWORK_TOOLREGISTRY_H

#include "../interfaces/IToolRegistry.h"

class ToolRegistry : public IToolRegistry {
private:
    std::unordered_map<std::string, std::shared_ptr<ITool>> tools;
public:
    void register_tool(std::shared_ptr<ITool> tool) override;
    std::shared_ptr<ITool> get_tool(const std::string & name) const override;
    bool has_tool(const std::string & name) const override;
    void remove_tool(const std::string & name) override;

    std::vector<std::string> get_tool_names() const override;
    std::vector<ToolDefinition> get_all_definitions() const override;

    ToolResult execute_tool(const std::string & name, const json & arguments) const override;
    std::string format_tools_for_prompt() const override;
};


#endif //AGENT_FRAMEWORK_TOOLREGISTRY_H