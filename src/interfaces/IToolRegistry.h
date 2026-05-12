#ifndef AGENT_FRAMEWORK_ITOOLREGISTRY_H
#define AGENT_FRAMEWORK_ITOOLREGISTRY_H

#include <string>
#include <vector>
#include <memory>
#include "ITool.h"

class IToolRegistry {
public:
    virtual ~IToolRegistry() = default;

    virtual void register_tool(std::shared_ptr<ITool> tool) = 0;
    virtual std::shared_ptr<ITool> get_tool(const std::string & name) const = 0;
    virtual bool has_tool(const std::string & name) const = 0;
    virtual void remove_tool(const std::string & name) = 0;

    virtual std::vector<std::string> get_tool_names() const = 0;
    virtual std::vector<ToolDefinition> get_all_definitions() const = 0;

    virtual ToolResult execute_tool(const std::string & name, const json & arguments) const = 0;

    virtual std::string format_tools_for_prompt() const = 0;
};



#endif //AGENT_FRAMEWORK_ITOOLREGISTRY_H