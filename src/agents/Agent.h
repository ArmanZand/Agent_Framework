#ifndef AGENT_FRAMEWORK_AGENT_H
#define AGENT_FRAMEWORK_AGENT_H

#include "../infrastructure/ContextManager.h"
#include "../model_services/ModelService.h"
#include "../model_services/ContextService.h"
#include "../model_services/SamplerService.h"
#include "../model_services/PromptService.h"
#include "../model_services/EmbeddingService.h"
#include "../infrastructure/ToolConfig.h"
#include "../tools/ToolRegistry.h"

#include "../rag/knowledge/KnowledgeBase.h"
#include "../rag/skills/SkillStore.h"
#include "../rag/memory/MemoryManager.h"

#include "../infrastructure/AgentStats.h"
#include "StreamProcessor.h"


namespace RoleType {
    inline constexpr const char * System    = "system";
    inline constexpr const char * User      = "user";
    inline constexpr const char * Assistant = "assistant";
    inline constexpr const char * Tool     = "tool";
}


using ToolStartCallback = std::function<void(const ToolCall & call)>;
using ToolResultCallback = std::function<void(const ToolCall & call, const ToolResult & result)>;
using ThinkingStartCallback = std::function<void()>;
using ThinkingEndCallback = std::function<void()>;


struct AgentConfig {
    ContextConfig context;
    SamplerConfig sampler;
    PromptConfig prompt;
    SummarizationConfig summarization;
    ToolConfig tools;
    AgentCallbacks callbacks;

    std::string name = "Agent";

    VocabularyConfig vocabulary;

    struct {
        float hint_min_score = 0.7;
        int hint_top_k = 5;
        int compaction_token_threshold = 150;
        std::string directory_path = ".";
    } knowledge;

    struct {
        float hint_min_score = 0.65f;
        int hint_top_k = 5;
        int midturn_result_min_len = 200;
        int consolidate_assistant_threshold = 80;
        std::string memory_file_path = "agent_memory.json";
    } memory;


    struct {
        std::vector<std::string> skill_ids_to_set;
        std::string directory_path = ".";
    } skills;

    bool verbose = false;
};

struct GenerationPass {
    GenerationResult result;
    std::string response;
};

class Agent {
public:
    struct PrivateToken {
    private:
        PrivateToken() = default;
        friend class Agent;
    };
    explicit Agent(
        PrivateToken,
        const AgentConfig &config,
        std::shared_ptr<ModelService> model,
        std::shared_ptr<ContextService> context,
        std::shared_ptr<SamplerService> sampler,
        std::shared_ptr<PromptService> prompt,
        std::shared_ptr<SummarizationService> summarizer,
        std::shared_ptr<EmbeddingService> embedder);
    ~Agent();

    Agent(const Agent&) = delete;
    Agent& operator=(const Agent&) = delete;
    static std::shared_ptr<Agent> create(const AgentConfig & config, std::shared_ptr<ModelService> model, std::shared_ptr<EmbeddingService> embedder);

    std::string chat(const std::string & user_message);
    void new_conversation();


    int context_usage() const { return context_->token_count();}
    int context_size() const { return context_->allocated_context(); }


    void set_role_description(const std::string & role_description);
    void set_tools(std::vector<std::shared_ptr<ITool>> tools);
    void set_skills(const std::vector<std::string> & skill_ids);
    void set_knowledge_directory(const std::string & path);
    void set_skill_directory(const std::string & path);
    void set_memory_file(const std::string & path);
    void save_memory();

    void clear_skills();
    std::string build_system_prompt();

    ContextStats context_stats() const { return context_manager_.get_stats(); };
    AgentStats stats() const { return stats_; }
private:
    //Constructor order
    AgentConfig config_;

    std::shared_ptr<ModelService> model_;
    std::shared_ptr<ContextService> context_;
    std::shared_ptr<SamplerService> sampler_;
    std::shared_ptr<PromptService> prompt_;
    std::shared_ptr<SummarizationService> summarizer_;
    std::shared_ptr<EmbeddingService> embedder_;

    ContextManager context_manager_;
    //Chat loop
    void begin_turn(TurnStats & ts);
    std::string build_enriched_user_message(const std::string & user_message);
    void run_generation_loop(StreamState & state, std::string & accumulated, std::string & final_response, TurnStats & ts);
    void finalize_turn(const std::string & user_message, const std::string & final_response, TurnStats & ts, double turn_ms);
    std::unique_ptr<StreamProcessor> stream_processor_;

    //System
    std::string agent_role_description_;
    std::string agent_tools_description_;
    bool init_rag_components();

    //Skills
    std::shared_ptr<SkillStore> skill_store_;
    std::unordered_set<std::string> active_skills_;

    //Knowledge
    std::shared_ptr<KnowledgeBase> knowledge_base_;

    //Memory
    std::shared_ptr<MemoryManager> memory_manager_;
    std::string memory_path_;
    void consolidate_memory(const std::string & user_message, const std::string & assistant_response);
    std::string build_memory_hints(const std::string & query) const;
    std::string memory_enrich_user_message(const std::string & user_message);
    //Generation
    ToolRegistry tool_registry_;
    GenerationPass run_generation_pass(StreamState & state);
    bool run_tool_calls(const StreamState & state, const std::string & final_response, TurnStats & stats);
    //Context

    bool recover_from_context_full(StreamState & state, std::string & accumulated, int & count, TurnStats & stats);



    //Stats
    AgentStats stats_;
};


#endif //AGENT_FRAMEWORK_AGENT_H