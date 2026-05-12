#include "RememberTool.h"

RememberTool::RememberTool(std::shared_ptr<MemoryManager> memory): memory_(std::move(memory)) {}

ToolDefinition RememberTool::get_definition() const {
    ToolDefinition td;
    td.name = "remember",
            td.description =
            "Store a fact or discovery in memory for future reference. "
            "Use mid-task to record important findings so they can be recalled "
            "later without re-reading source material. ";
    td.parameters = {
        {"fact", ToolArgType::String, "A single self-contained sentence to remember", true}
    };
    return td;
}

ToolResult RememberTool::execute(const json &args) {
    std::string fact = args.value("fact", "");
    if (fact.empty()) return { false, {}, "fact is required"};
    return { true, memory_->insert(fact) ? "Remembered." : "Already known." };
}
