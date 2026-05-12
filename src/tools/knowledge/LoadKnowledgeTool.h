#ifndef AGENT_FRAMEWORK_LOADKNOWLEDGETOOL_H
#define AGENT_FRAMEWORK_LOADKNOWLEDGETOOL_H

#include <memory>
#include "../../interfaces/ITool.h"
#include "../../rag/knowledge/KnowledgeBase.h"

class LoadKnowledgeTool : public ITool {
public:
    explicit LoadKnowledgeTool(std::shared_ptr<KnowledgeBase> kb);
    ToolDefinition get_definition() const override;
    ToolResult execute(const json & args) override;
private:
    std::shared_ptr<KnowledgeBase> kb_;
};


#endif //AGENT_FRAMEWORK_LOADKNOWLEDGETOOL_H