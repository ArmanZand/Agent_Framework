#ifndef AGENT_FRAMEWORK_EXPLOREDIRECTORYTOOL_H
#define AGENT_FRAMEWORK_EXPLOREDIRECTORYTOOL_H
#include <string>

#include "../../interfaces/ITool.h"

class ExploreDirectoryTool : public ITool {
private:
    std::string base_directory;
public:
    ExploreDirectoryTool(const std::string & base_dir = ".");
    ToolDefinition get_definition() const override;
    ToolResult execute(const json &arguments) override;
};

#endif //AGENT_FRAMEWORK_EXPLOREDIRECTORYTOOL_H