#include "ListKnowledgeTool.h"

ListKnowledgeTool::ListKnowledgeTool(std::shared_ptr<KnowledgeBase> kb) : kb_(std::move(kb))
{}

ToolDefinition ListKnowledgeTool::get_definition() const {
    return {
        "list_knowledge",
        "List the knowledge cards in a given topic, optionally filtered by a keyword query. "
        "Returns each card's id, title, and one-line summary. "
        "Use this to browse what's available before loading a card.",
        {
            {
                "topic", ToolArgType::String,
                "The topic to list (e.g. 'topic' or 'topic/subtopic'). "
                "Use empty string '' to list all cards across all topics.",
                true
            },
            {
                "query", ToolArgType::String,
                "Optional keyword filter applied within the topic.",
                false
            }
        }
    };
}

ToolResult ListKnowledgeTool::execute(const json &args) {
    ToolResult r;
    if (!kb_) {
        r.success = false;
        r.error = "Knowledge base is not initialized.";
        return r;
    }
    std::string topic = args.value("topic", "");
    std::string query = args.value("query", "");
    auto cards = kb_->list(topic, query);
    std::ostringstream os;
    if (cards.empty()) {
        os << "No cards found";
        if (!topic.empty()) os << " in topic '" << topic << "'";
        if (!query.empty()) os << " matching '" << query << "'";
        os << ".";
    } else {
        os << cards.size() << " cards(s):\n";
        for (auto & c : cards) {
            os << "- " << c.id << " — " << c.title;
            if (!c.summary.empty()) os << ": " << c.summary;
            os << "\n";
        }
    }
    r.success = true;
    r.result = os.str();
    return r;
}
