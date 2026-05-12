#ifndef AGENT_FRAMEWORK_REMEMBERTOOL_H
#define AGENT_FRAMEWORK_REMEMBERTOOL_H

#include "../../interfaces/ITool.h"
#include "../../rag/memory/MemoryManager.h"

class RememberTool : public ITool {
public:
    explicit RememberTool(std::shared_ptr<MemoryManager> memory);
    ToolDefinition get_definition() const override;
    ToolResult execute(const json &  args) override;
private:
    std::shared_ptr<MemoryManager> memory_;
};


#endif //AGENT_FRAMEWORK_REMEMBERTOOL_H