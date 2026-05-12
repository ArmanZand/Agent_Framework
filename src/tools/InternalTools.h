#ifndef AGENT_FRAMEWORK_INTERNALTOOLS_H
#define AGENT_FRAMEWORK_INTERNALTOOLS_H
#include "../interfaces/ITool.h"
#include <functional>
#include <memory>
#include <string>


class CustomFunctionTool : public ITool {
private:
    ToolDefinition definition;
    std::function<ToolResult(const std::string&)> tool_function;
public:
    CustomFunctionTool(ToolDefinition definition_, std::function<ToolResult(const std::string&)> function_) :
     definition(std::move(definition_)), tool_function(std::move(function_)) {};

    ToolDefinition get_definition() const override { return definition; };
    ToolResult execute(const json &arguments) override { return tool_function(arguments); };
};

#endif //AGENT_FRAMEWORK_INTERNALTOOLS_H