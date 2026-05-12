#include "SearchMemoryTool.h"

SearchMemoryTool::SearchMemoryTool(std::shared_ptr<MemoryManager> memory): memory_(std::move(memory)) {}

ToolDefinition SearchMemoryTool::get_definition() const {
    ToolDefinition td;
    td.name =  "search_memory";
    td.description =
            "Search memory for facts and information stored from prior sessions "
            "or earlier in this conersation. Use when you need to recall something specific.";
    td.parameters = {
        { "query", ToolArgType::String, "What to search for", true},
        {"top_k", ToolArgType::Number, "Max results (default 3)", false}
    };
    return td;
}

ToolResult SearchMemoryTool::execute(const json &args) {
    std::string query = args.value("query", "");
    int top_k = args.value("top_k", 3);
    if (query.empty()) return { false, {}, "query is required"};
    auto hits = memory_->search(query, top_k);
    if (hits.empty()) return { true, "No relevant memories found."};
    std::string out;
    for (auto & h : hits)
        out += "- [" + std::to_string(static_cast<int>(h.score * 100)) + "%] " + h.data + "\n";
    return { true, out };
}
