#ifndef AGENT_FRAMEWORK_FILEWRITETOOL_H
#define AGENT_FRAMEWORK_FILEWRITETOOL_H

#include "../../interfaces/ITool.h"
#include <string>

class FileWriteTool : public ITool {
private:
    std::string base_directory;
public:
    FileWriteTool(const std::string & base_dir = ".");
    ToolDefinition get_definition() const override;
    ToolResult execute(const json &arguments) override;
};

#endif //AGENT_FRAMEWORK_FILEWRITETOOL_H