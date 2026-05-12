#include "SearchKnowledgeTool.h"

SearchKnowledgeTool::SearchKnowledgeTool(std::shared_ptr<KnowledgeBase> kb): kb_(std::move(kb)) {}

ToolDefinition SearchKnowledgeTool::get_definition() const {
    return {
        "search_knowledge",
        "Find knowledge cards by keyword across all topics. "
        "Returns the top matching cards as id, title, and summary. "
        "Use this when you don't know which topic the answer is in.",
        {
            {
                "query", ToolArgType::String,
                "Keywords describing what you're looking for.",
                true
            }
        }
    };
}

ToolResult SearchKnowledgeTool::execute(const json &args) {
    ToolResult r;
    if (!kb_) {
        r.success = false;
        r.error = "Knowledge base is not initialized.";
        return r;
    }
    if (!args.contains("query") || !args["query"].is_string()) {
        r.success = false;
        r.error = "Missing required string argument 'query'.";
        return r;
    }
    std::string q = args["query"].get<std::string>();
    auto hits = kb_->search(q, 10);
    std::ostringstream os;
    if (hits.empty()) {
        os << "No cards matched '" << q << "'.";
    } else {
        os << hits.size() << " match(es) for '" << q << "':\n";
        for (auto & h : hits) {
            os << "- " << h.id << " — " << h.title;
            if (!h.summary.empty()) os << ": " << h.summary;
            os << "\n";
        }
    }
    r.success = true;
    r.result = os.str();
    return r;
}

