/**
 * @file Agent.h
 * @brief Contains the core agent interface that combines model services and other components to produce responses.
 */

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

/**
 * @namespace RoleType
 * @brief Conversation role identifiers that can be referenced as a type.
 */
namespace RoleType {
    /** @brief System prompt identifier. */
    inline constexpr const char * System    = "system";
    /** @brief User prompt identifier. */
    inline constexpr const char * User      = "user";
    /** @brief Assistant response identifier. */
    inline constexpr const char * Assistant = "assistant";
    /** @brief Tool call identifier. */
    inline constexpr const char * Tool     = "tool";
}

/**
 * @brief Runtime configuration for an Agent instance.
 *
 * Contains configuration for services such as context, sampler, prompt, summarization.
 * The agent can also be configured to format its response in certain ways by altering the grammar in the vocabulary configuration.
 * Also contains parameters for skills, knowledge, memory and hints.
 */
struct AgentConfig {
    ContextConfig context; /** @brief Context */
    SamplerConfig sampler;
    PromptConfig prompt;
    SummarizationConfig summarization;
    ToolConfig tools;
    AgentCallbacks callbacks;
    VocabularyConfig vocabulary;

    std::string name = "Agent";

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

/** @brief Result container for a singler generation pass. */
struct GenerationPass {
    GenerationResult result;
    std::string response;
};

/**
 * @brief Conversational agent with tool, memory and retrieval support components.
 *
 * The agent combines model services, token streaming and parsing with tool calls to respond to user queries.
 * Keeps track of conversation statistics including context state information.
 */
class Agent {
public:
    /** @brief Private token to aid Agent::create access private members to initialize the agent state. */
    struct PrivateToken {
    private:
        PrivateToken() = default;
        friend class Agent;
    };


    /**
     * @brief Constructs an Agent instance with the required model services.
     *
     * Initializes the model services required for the agent to generate responses.
     * PrivateToken is used to allow access to private members for Agent::create which builds and defines the services used for the
     * constructor.
     *
     * Use Agent::create to initialize an agent instance.
     *
     * @param config Agent runtime configuration.
     * @param model Shared language model service to be used for responses or summarization with their own contexts.
     * @param context Agent context service to interface with kv-cache.
     * @param sampler Agent token sampling service.
     * @param prompt Agent prompt generation service.
     * @param summarizer Shared content summarization service.
     * @param embedder Shared embedder service with its own model for retrieval and memory searching.
     */
    explicit Agent(
        PrivateToken,
        const AgentConfig &config,
        std::shared_ptr<ModelService> model,
        std::shared_ptr<ContextService> context,
        std::shared_ptr<SamplerService> sampler,
        std::shared_ptr<PromptService> prompt,
        std::shared_ptr<SummarizationService> summarizer,
        std::shared_ptr<EmbeddingService> embedder);

    /**
     * @brief Agent destructor saves memory state.
     *
     * State of the memory manager is saved.
     */
    ~Agent();

    Agent(const Agent&) = delete;
    Agent& operator=(const Agent&) = delete;

    /**
     * @brief Creates and initializes a configured Agent instance.
     *
     * Starts the required Agent services using the given agent configuration with summarization, embedding and RAG support.
     *
     * @param config Agent configuration.
     * @param model Shared language model service.
     * @param embedder Shared embedding service.
     * @return Shared pointer to a valid Agent instance or nullptr on failure.
     */
    static std::shared_ptr<Agent> create(const AgentConfig & config, std::shared_ptr<ModelService> model, std::shared_ptr<EmbeddingService> embedder);

    /**
     * @brief Processes a user message and generates a response.
     *
     * Performs a conversation turn from a user message which can use callbacks to stream tokens and states such as
     * thinking and tool calls.
     *
     * @param user_message User input message for the model to respond to.
     * @return Final assistant response.
     */
    std::string chat(const std::string & user_message);

    /**
     * @brief Starts a new conversation with the current agent configuration.
     *
     * Clears context and conversation history.
     */
    void new_conversation();


    int context_usage() const { return context_->token_count();}
    int context_size() const { return context_->allocated_context(); }

    /**
     * @brief Manually set the agent system prompt description.
     *
     * Rebuilds the context with the new description.
     *
     * @param role_description Agent system prompt.
     */
    void set_role_description(const std::string & role_description);


    /**
     * @brief Registers tools in the Agent's registry.
     *
     * Sets the set of tools to be available to agent. If any tools are present the system prompt will include instructions
     * to perform tool calling.
     *
     * @param tools Set of tool instances to register.
     */
    void set_tools(std::vector<std::shared_ptr<ITool>> tools);

    /**
     * @brief Enables a set of skills to be loaded in the context by their identifiers.
     *
     * Searches the skill store for the respective identifiers and loads them into the agent context after the system prompt.
     * @param skill_ids Set of skill identifiers to activate.
     */
    void set_skills(const std::vector<std::string> & skill_ids);

    /**
     * @brief Loads knowledge cards from a given directory, acting as a knowledge-base.
     *
     * Loads the knowledge-base by reading knowledge cards in the directory. If the knowledge-base is loaded then
     * complimentary tools are registered to aid the model in searching and loading knowledge-cards.
     * These cards can also appear as mid-turn hints to remind the model of existing relevant information.
     *
     * @param path Path to the directory containing knowledge cards.
     */
    void set_knowledge_directory(const std::string & path);

    /**
     * @brief Loads agent skills from a directory.
     *
     * Parses agent skills in the directory allowing the agent to enable a selection by their identifiers.
     * @param path Path to the directory containing skill files.
     */
    void set_skill_directory(const std::string & path);


    /**
     * @brief Sets the file to load or save the state of the memory manager.
     *
     * Loads or saves JSON to the file.
     * @param path Path to JSON file.
     */
    void set_memory_file(const std::string & path);

    /**
     * @brief Saves the state of the memory manager.
     *
     * Saves the state to the set memory file.
     */
    void save_memory();

    /**
     * @brief Removes the current active skills from the agent context.
     */
    void clear_skills();

    /**
     * @rbief Builds and injects the current system prompt.
     *
     * Combines role description, behaviour instructions, tool description, knowledge guidance and memory guidance
     * into a single system prompt string.
     * @return Constructed system prompt string.
     */
    std::string build_system_prompt();

    /**
     * @brief Obtains struct containing stats of the current state of the agent context.
     * @return Context stats struct.
     */
    ContextStats context_stats() const;

    /**
     * @brief Obtains the current overall stats related to agent state and generation results.
     * @return Agent stats struct.
     */
    AgentStats stats() const;

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
    /**
     * @brief Initializes the agent chat statistics for the beginning of the turn.
     *
     * @param ts Turn statistics instance.
     */
    void begin_turn(TurnStats & ts);

    /**
     * @brief Enriches the user message with memory and knowledge hints.
     *
     * Retrieves relevant memory and knowledge search results and prepends them to the user message for improved agent
     * context.
     *
     * @param user_message Original user input.
     * @return Enriched user message.
     */
    std::string build_enriched_user_message(const std::string & user_message);

    /**
     * @brief Runs the main response generation with a tool calling loop.
     *
     * Handle response generation, tool execution, context recovery and returning a final response.
     *
     * @param state Stream state for callback procs.
     * @param accumulated Accumulated partial responses.
     * @param final_response Final assistant response.
     * @param ts Turn stats instance.
     */
    void run_generation_loop(StreamState & state, std::string & accumulated, std::string & final_response, TurnStats & ts);

    /**
     * @brief Finalizes a chat turn after executing the main generation loop.
     *
     * Updates the conversation history and turn statistics. Consolidates obtain information during the turn
     * by compacting tool results and extracting facts obtained.
     *
     * @param user_message Original user message.
     * @param final_response Final assistant response.
     * @param ts Turn stats instance.
     * @param turn_ms Total trn execution time in milliseconds.
     */
    void finalize_turn(const std::string & user_message, const std::string & final_response, TurnStats & ts, double turn_ms);

    std::unique_ptr<StreamProcessor> stream_processor_;

    //System
    std::string agent_role_description_;
    std::string agent_tools_description_;

    /**
     * @brief Initializes RAG components with agent config settings.
     *
     * Uses agent config settings to set up skills, knowledge-base and memory.
     *
     * @return  True if initialization succeeds, otherwise false.
     */
    bool init_rag_components();

    //Skills
    std::shared_ptr<SkillStore> skill_store_;
    std::unordered_set<std::string> active_skills_;

    //Knowledge
    std::shared_ptr<KnowledgeBase> knowledge_base_;

    //Memory
    std::shared_ptr<MemoryManager> memory_manager_;
    std::string memory_path_;

    /**
     * @brief Extracts and stores long-term facts from the conversation turn.
     *
     * Uses summarization service to extract atomic facts and inserts non-duplicate entries into the memory manager.
     *
     * @param user_message User input message.
     * @param assistant_response Assistant response from turn.
     */
    void consolidate_memory(const std::string & user_message, const std::string & assistant_response);

    /**
     * @brief Retrieves relevant information related to a query and builds hints to be injected in the context.
     * @param query Query to search in the memory manager.
     * @return Formatted hint text block.
     */
    std::string build_memory_hints(const std::string & query) const;

    /**
     * @brief Recalls memory entries as hints that may be relevant to the user message.
     * @param user_message Original user message.
     * @return Memory hint text block.
     */
    std::string memory_enrich_user_message(const std::string & user_message);
    //Generation
    ToolRegistry tool_registry_;

    /**
     * @brief Executes a single generation pass.
     * Streams tokens through the stream processor and returns the generated response state.
     * @param state Stream state containing tool calls and callbacks.
     * @return Generation pass result struct.
     */
    GenerationPass run_generation_pass(StreamState & state);

    /**
     * @brief Executes completed tool calls from the current generation pass.
     * @param state Stream state instance.
     * @param final_response Agent final response.
     * @param stats Turn stats instance.
     * @return True if all tools executed successfully, False otherwise.
     */
    bool run_tool_calls(const StreamState & state, const std::string & final_response, TurnStats & stats);

    //Context
    /**
     * @brief Attempts to make room in the agent context when it becomes full.
     * @param state Stream state instance.
     * @param accumulated Accumulated partial responses.
     * @param count Number of times context full was encountered.
     * @param stats Turn stats instance.
     * @return True if recovery succeeded, False otherwise.
     */
    bool recover_from_context_full(StreamState & state, std::string & accumulated, int & count, TurnStats & stats);

    //Stats
    AgentStats stats_;
};


#endif //AGENT_FRAMEWORK_AGENT_H