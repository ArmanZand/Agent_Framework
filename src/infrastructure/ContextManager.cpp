#include "ContextManager.h"

ContextManager::ContextManager(std::shared_ptr<ContextService> context, std::shared_ptr<PromptService> prompt,
    std::shared_ptr<SummarizationService> summarization) :
context_(context),
prompt_(prompt),
summarizer_(summarization)
{
    rolling_summary_ = { MessageRole::Summary, "", "rolling_summary", true, 0, 0, false };
}

void ContextManager::add_system(const std::string &content) {
    system_ = { MessageRole::System, content, "system", false, 0, 0, false };
    cache_dirty_ = true;
}

void ContextManager::add_skill(const std::string &id, const std::string &content) {
    for (auto & s : skills_) {
        if (s.id == id) { s.content = content; cache_dirty_ = true; return; }
    }
    skills_.push_back({ MessageRole::Skill, content, id, true, 0, 0, false });
    cache_dirty_ = true;
}

void ContextManager::remove_skill(const std::string &id) {
    auto it = std::remove_if (skills_.begin(), skills_.end(), [&](const ManagedMessage & m) { return m.id == id; });
    if (it != skills_.end()) {
        skills_.erase(it , skills_.end());
        cache_dirty_ = true;
    }
}

void ContextManager::clear_skills() {
    skills_.clear();
    cache_dirty_ = true;
}

void ContextManager::add_user(const std::string &content) {
    tail_.push_back({MessageRole::User, content, "", false, 0, 0, false});
}

void ContextManager::add_assistant(const std::string &content) {
    tail_.push_back({MessageRole::Assistant, content, "", false, 0, 0, false});
}

void ContextManager::add_assistant_partial(const std::string &content) {
    for (auto & m : tail_) m.partial = false;
    tail_.push_back({MessageRole::Assistant, content, "", false, 0, 0, false });
    tail_.back().partial = true;
}

void ContextManager::add_tool_call(const ToolCall &call) {
    std::string content = "<tool_call>" + call.arguments.dump() + "</tool_call>";
    tail_.push_back({MessageRole::Assistant, content, call.id, false, 0, 0 ,false});
}

bool ContextManager::add_tool_result(const ToolCall &call, const ToolResult &result, bool is_last) {
    std::string content = result.success ? result.result : ("Error: " + result.error);
    ManagedMessage msg { MessageRole::Tool, content, call.id, false, 0,0, false};
    tail_.push_back(msg);
    return encode_message(tail_.back(), is_last);
}

void ContextManager::compact_tool_results(int token_threshold) {
    if (token_threshold <= 0) return;

    int last_assistant = -1;
    for (int i = (int)tail_.size() - 1; i >= 0; --i) {
        if (tail_[i].role == MessageRole::Assistant) {
            last_assistant = i;
            break;
        }
    }
    if (last_assistant <= 0) return;

    bool any = false;
    for (int i = 0; i < last_assistant; ++i) {
        auto & msg = tail_[i];
        if (msg.role != MessageRole::Tool) continue;;
        std::string rendered = prompt_->render_turn(msg.role, msg.content, false);
        int n = (int)prompt_->tokenize(rendered, false).size();
        if (n <= token_threshold) continue;
        fprintf(stdout, "[ContextManager] Compacting tool result '%s' (%d tokens -> stub).\n",
                msg.id.c_str(), n);
        msg.content  = "[Tool result compacted to free context. The data was processed in the preceding response. Avoid re-calling this tool unless necessary.]";
        msg.encoded = false;
        any = true;
        ++stat_results_stubbed_;
    }
    if (any) cache_dirty_ = true;

}

ContextStats ContextManager::get_stats() const {
    ContextStats s;
    s.capacity = (int)context_->allocated_context();
    s.used = (int)context_->token_count();
    s.remaining = (int)context_->remaining();

    auto count_tokens = [&](const ManagedMessage & m) -> int {
      if (m.content.empty()) return 0;
        return (int)prompt_->tokenize(prompt_->render_turn(m.role, m.content, false), false).size();
    };

    s.system_tokens = count_tokens(system_);
    s.summary_tokens = count_tokens(rolling_summary_);
    for (const auto & sk : skills_)
        s.skills_tokens += count_tokens(sk);

    for (const auto & m : tail_) {
        int t = count_tokens(m);
        s.tail_tokens += t;
        s.tail_messages++;
        if (m.role == MessageRole::Tool) s.tail_tool_tokens += t;
        else if (m.role == MessageRole::Assistant) s.tail_assistant_tokens += t;
        else if (m.role == MessageRole::User) s.tail_user_tokens += t;
    }
    for (const auto & m : archive_) {
        s.archive_tokens += count_tokens(m);
        s.archive_messages++;
    }
    return s;
}

void ContextManager::clear_history() {
    tail_.clear();
    archive_.clear();
    rolling_summary_.content.clear();
    rolling_summary_.encoded = false;
    cache_dirty_ = true;
}



std::vector<ManagedMessage *> ContextManager::render_order() {
    std::vector<ManagedMessage*> order;
    if (!system_.content.empty())
        order.push_back(&system_);
    for (auto & s : skills_)
        order.push_back(&s);
    if (!rolling_summary_.content.empty())
        order.push_back(&rolling_summary_);
    for (auto & m : archive_)
        order.push_back(&m);
    for (auto & m : tail_)
        order.push_back(&m);
    return order;
}

void ContextManager::rebuild_cache() {
    rebuilding_ = true;
    context_->clear_kv_cache();

    system_.encoded = false;
    rolling_summary_.encoded = false;
    for (auto & s : skills_) s.encoded = false;
    for (auto & m : archive_) m.encoded = false;
    for (auto & m : tail_) m.encoded = false;

    auto order = render_order();
    for (size_t i = 0; i < order.size(); ++i) {
        bool is_last =  (i == order.size() - 1);
        if (!encode_message(*order[i], is_last)) {
            fprintf(stderr, "[ContextManager] Failed to encode message during rebuild.\n");
            break;
        }
    }
    cache_dirty_ = false;
    rebuilding_ = false;
}

bool ContextManager::flush_pending() {
    if (cache_dirty_) {
        rebuild_cache();
        return true;
    }

    auto order = render_order();
    for (size_t i = 0; i < order.size(); ++i) {
        if (!order[i]->encoded) {
            bool is_last = (i == order.size() - 1);
            if (!encode_message(*order[i], is_last)) {
                return false;
            }
        }
    }
    return true;
}

void ContextManager::tick() {
    int total = context_->allocated_context();
    int available = total - fixed_token_count() - reserved_response_tokens_;
    if (available < 0) available = 0;

    while ((int)tail_.size() > min_keep_messages_ && tail_token_count() > available) {
        archive_.push_back(std::move(tail_.front()));
        tail_.erase(tail_.begin());
    }
    if (!archive_.empty()) {
        compress_archive();
    }
    ++stat_tick_events_;
}


bool ContextManager::encode_message(ManagedMessage &message, bool add_generation_prompt) {
    std::string rendered;
    if (message.partial && add_generation_prompt) {
        rendered = prompt_->render_partial_assistant(message.content);
    }else {
        if (message.partial) message.partial = false;
        rendered = prompt_->render_turn(message.role, message.content, add_generation_prompt);
    }
    auto tokens = prompt_->tokenize(rendered, false);
    if (!rebuilding_ && tokens.size() >= context_->remaining()) {
        fprintf(stderr, "[ContextManager] Not enough space for message (need %zu, have %u).\n", tokens.size(), context_->remaining());
        tick();
        rendered = prompt_->render_turn(message.role, message.content, add_generation_prompt);
        tokens = prompt_->tokenize(rendered, false);
        if (tokens.size() >= context_->remaining()) {
            fprintf(stderr, "[ContextManager] Still not enough space after compression (need %zu, have %u).\n", tokens.size(), context_->remaining());
            return false;
        }
    }

    message.kv_start = context_->token_count();
    if (prompt_->decode_batch(*context_, tokens) != 0) {
        fprintf(stderr, "[ContextManager] decode_batch failed.\n");
        return false;
    }
    message.kv_end = context_->token_count();
    message.encoded = true;
    return true;

}

void ContextManager::compress_archive() {
    if (archive_.empty()) return;

    std::vector<ChatMessage> to_summarize;
    if (!rolling_summary_.content.empty()) {
        to_summarize.push_back({MessageRole::Summary, rolling_summary_.content});
    }
    for (auto & m : archive_) {
        to_summarize.push_back({m.role, m.content});
    }
    std::string new_summary = summarizer_->summarize(to_summarize);

    if (!new_summary.empty()) {
        rolling_summary_.content = new_summary;
        rolling_summary_.encoded = false;
    }

    archive_.clear();
    cache_dirty_ = true;
    rebuild_cache();

    if (verbose_) {
        fprintf(stdout, "[ContextManager] Rolling summary updated (%zu chars):\n%s\n",
            rolling_summary_.content.size(), rolling_summary_.content.c_str());
    }
    ++stat_compression_;
}

int ContextManager::tail_token_count() const {
int total = 0;
    for (const auto & m : tail_) {
        std::string rendered = prompt_->render_turn(m.role, m.content, false);
        total += (int)prompt_->tokenize(rendered, false).size();
    }
    return total;
}

int ContextManager::fixed_token_count() const {
    int total = 0;
    auto count = [&](const ManagedMessage & m) {
        if (m.content.empty()) return;
        std::string rendered = prompt_->render_turn(m.role, m.content, false);
        total += (int)prompt_->tokenize(rendered, false).size();
    };
    count(system_);
    count(rolling_summary_);
    for (const auto & s : skills_) count(s);
    return total;
}


