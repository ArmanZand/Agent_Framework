#include "LoadKnowledgeTool.h"

LoadKnowledgeTool::LoadKnowledgeTool(std::shared_ptr<KnowledgeBase> kb)
: kb_(std::move(kb))
{}

ToolDefinition LoadKnowledgeTool::get_definition() const {
    return {
        "load_knowledge",
        "Load the full content of a knowledge card by its id. "
        "A knowledge card is an object that contains curated text on a particular topic or domain. "
        "Use after list_knowledge or search_knowledge surfaces a relevant card. "
        "Returns the card body wrapped with grounding instructions.",
        {
            {
                "id", ToolArgType::String,
                "The card id (e.g. 'black-scholes'). Must match an id from the manifest.",
                true
            }
        }

    };
}

ToolResult LoadKnowledgeTool::execute(const json &args) {
    ToolResult r;
    if (!kb_) {
        r.success = false;
        r.error = "Knowledge base is not initialized.";
        return r;
    }
    if (!args.contains("id") || !args["id"].is_string()) {
        r.success = false;
        r.error = "Missing required  string argument 'id'.";
        return r;
    }
    std::string id = args["id"].get<std::string>();
    auto body = kb_->load_body(id);
    if (!body) {
        r.success = false;
        r.error = "No card with id '" + id + "' (or body could not be read).";
        return r;
    }
    std::ostringstream os;
    os << "<knowledge id=\"" << id << "\">\n"
       << *body
       << "\n</knowledge>\n"
       << "Use the above knowledge to answer the user's question. "
       << "If it does not fully cover the question, say so explicitly rather than guessing.";
    r.success = true;
    r.result = os.str();
    return r;
}
