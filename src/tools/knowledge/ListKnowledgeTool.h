#ifndef AGENT_FRAMEWORK_LISTKNOWLEDGETOOL_H
#define AGENT_FRAMEWORK_LISTKNOWLEDGETOOL_H

#include <memory>
#include "../../interfaces/ITool.h"
#include "../../rag/knowledge/KnowledgeBase.h"

class ListKnowledgeTool : public ITool {
public:
    explicit ListKnowledgeTool(std::shared_ptr<KnowledgeBase> kb);
    ToolDefinition get_definition() const override;
    ToolResult execute(const json & args) override;
private:
    std::shared_ptr<KnowledgeBase> kb_;

};


#endif //AGENT_FRAMEWORK_LISTKNOWLEDGETOOL_H