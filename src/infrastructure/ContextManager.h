#ifndef AGENT_FRAMEWORK_CONTEXTMANAGER_H
#define AGENT_FRAMEWORK_CONTEXTMANAGER_H
#include <memory>
#include <string>
#include "../model_services/ContextService.h"
#include "../model_services/PromptService.h"
#include "../model_services/SummarizationService.h"
#include "../interfaces/ITool.h"
#include "../infrastructure/AgentStats.h"
#include "ChatMessage.h"


struct ManagedMessage : public ChatMessage {
    std::string id;
    bool evictable;
    uint32_t kv_start;
    uint32_t kv_end;
    bool encoded;
    bool partial = false;
};

class ContextManager {
public:
    ContextManager(
        std::shared_ptr<ContextService> context,
        std::shared_ptr<PromptService> prompt,
        std::shared_ptr<SummarizationService> summarization);

    void add_system(const std::string & content);
    void add_skill(const std::string & id, const std::string & content);
    void remove_skill(const std::string & id);
    void clear_skills();

    void add_user(const std::string & content);
    void add_assistant(const std::string & content);
    void add_assistant_partial(const std::string & content);
    void add_tool_call(const ToolCall & call);
    bool add_tool_result(const ToolCall & call, const ToolResult & result, bool is_last = false);
    void compact_tool_results(int token_threshold = 150);

    void rebuild_cache();
    bool flush_pending();
    void tick();

    uint32_t remaining() { return context_->remaining(); }
    uint32_t capacity() const { return context_->allocated_context(); }
    ManagedMessage rolling_summary() const { return rolling_summary_; }


    void clear_history();

    ContextStats get_stats() const;
    int stat_ticks() const { return stat_tick_events_; }
    int stat_compressions() const { return stat_compression_; }
    int stat_stubbed() const { return stat_results_stubbed_; }
    void reset_stats() {
        stat_tick_events_ = stat_compression_ = stat_results_stubbed_ = 0;
    }

private:
    std::vector<ManagedMessage *> render_order();
    bool encode_message(ManagedMessage & message, bool add_generation_prompt);
    void compress_archive();
    int tail_token_count() const;
    int fixed_token_count() const;

    std::shared_ptr<ContextService> context_;
    std::shared_ptr<PromptService> prompt_;
    std::shared_ptr<SummarizationService> summarizer_;

    std::vector<ManagedMessage> tail_;
    std::vector<ManagedMessage> archive_;
    ManagedMessage rolling_summary_;
    std::vector<ManagedMessage> skills_;
    ManagedMessage system_;

    bool cache_dirty_ = true;
    bool rebuilding_ = false;
    int min_keep_messages_ = 5;
    int reserved_response_tokens_ = 0;
    bool verbose_ = false;

    int stat_tick_events_ = 0;
    int stat_compression_ = 0;
    int stat_results_stubbed_ = 0;


};


#endif //AGENT_FRAMEWORK_CONTEXTMANAGER_H