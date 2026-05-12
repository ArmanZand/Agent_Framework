#ifndef AGENT_FRAMEWORK_FILEINFOTOOL_H
#define AGENT_FRAMEWORK_FILEINFOTOOL_H
#include <string>

#include "../../interfaces/ITool.h"

class FileInfoTool : public ITool {
private:
    std::string base_directory;
public:
    FileInfoTool(const std::string & base_dir = ".");
    ToolDefinition get_definition() const override;
    ToolResult execute(const json &arguments) override;
};



#endif //AGENT_FRAMEWORK_FILEINFOTOOL_H