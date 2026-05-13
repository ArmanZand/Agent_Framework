#include "Agent.h"

#include <sstream>
#include "../model_services/BackendService.h"
#include "../infrastructure/Stopwatch.h"
#include "../tools/knowledge/ListKnowledgeTool.h"
#include "../tools/knowledge/SearchKnowledgeTool.h"
#include "../tools/knowledge/LoadKnowledgeTool.h"
#include "../tools/memory/RememberTool.h"
#include "../tools/memory/SearchMemoryTool.h"
namespace fs = std::filesystem;


Agent::Agent(
    PrivateToken,
    const AgentConfig & config,
    std::shared_ptr<ModelService> model,
    std::shared_ptr<ContextService> context,
    std::shared_ptr<SamplerService> sampler,
    std::shared_ptr<PromptService> prompt,
    std::shared_ptr<SummarizationService> summarizer,
    std::shared_ptr<EmbeddingService> embedder) :
    config_(config),
    model_(std::move(model)),
    context_(context),
    sampler_(std::move(sampler)),
    prompt_(prompt),
    summarizer_(summarizer),
    embedder_(std::move(embedder)),
    context_manager_(std::move(context), std::move(prompt), std::move(summarizer))
{
    stream_processor_ = std::make_unique<StreamProcessor>(config_.vocabulary, config_.callbacks);
    fprintf(stdout, "[Agent] Initialized. Context: %u tokens\n", context_->allocated_context());
    fflush(stdout);
}


Agent::~Agent() {
    save_memory();
}

std::shared_ptr<Agent> Agent::create(const AgentConfig & config, std::shared_ptr<ModelService> model,
                                     std::shared_ptr<EmbeddingService> embedder) {
    auto & backend = BackendService::instance();
    if (!backend.is_running()) {
        if (!backend.start()) return nullptr;
    }
    auto context = std::make_shared<ContextService>(model, config.context);
    if (!context->initialize()) return nullptr;

    auto sampler = std::make_shared<SamplerService>(config.sampler);
    if (!sampler->initialize()) return nullptr;
    if (!config.vocabulary.grammar.empty()) {
        sampler->set_grammar(model->vocab(), config.vocabulary.grammar);
    }


    PromptConfig summarizer_config;
    summarizer_config.max_new_tokens = config.summarization.max_output_tokens;
    summarizer_config.get_default_template = true;
    auto summarizer_prompt = std::make_shared<PromptService>(model, summarizer_config);
    auto summarizer = std::make_shared<SummarizationService>(summarizer_prompt, model, config.summarization);

    auto prompt = std::make_shared<PromptService>(model, config.prompt);
    auto agent = std::make_shared<Agent>(PrivateToken{}, config, model, context, sampler, prompt, summarizer, embedder);


    if (!agent->init_rag_components()) return nullptr;

    return agent;
}


std::string Agent::chat(const std::string &user_message) {
    Stopwatch turn_sw;
    TurnStats ts;
    begin_turn(ts);

    std::string enriched = build_enriched_user_message(user_message);

    context_manager_.tick();
    context_manager_.add_user(enriched);

    if (!context_manager_.flush_pending()) {
        fprintf(stderr, "[Agent] Failed to flush context.\n");
    }

    StreamState state;
    std::string accumulated, final_response;
    run_generation_loop(state, accumulated, final_response, ts);

    finalize_turn(user_message, final_response, ts, turn_sw.ms());
    return accumulated.empty() ? final_response : accumulated + "\n\n" + final_response;
}


void Agent::new_conversation() {
    save_memory();
    context_->clear_kv_cache();
    context_manager_.clear_history();
    context_manager_.rebuild_cache();
}



void Agent::set_role_description(const std::string &role_description) {
    agent_role_description_ =role_description;
    build_system_prompt();
    context_manager_.flush_pending();
}

void Agent::set_tools(std::vector<std::shared_ptr<ITool>> tools) {
    for (auto & tool : tools) {
        tool_registry_.register_tool(tool);
    }
    if (tools.size() > 0) {
        agent_tools_description_ += "\n\nYou have access to the following tools:\n\n";
        agent_tools_description_ += tool_registry_.format_tools_for_prompt();
        agent_tools_description_ += "\n\nTo use a tool, respond with:\n";
        agent_tools_description_ += "<tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>\n\n";
        agent_tools_description_ += "You can use multiple tools by including multiple <tool_call> tags.";
    }

}

void Agent::set_skills(const std::vector<std::string> &skill_ids) {
    if (skill_ids.empty()) return;
    if (!skill_store_) {
        fprintf(stderr, "[Agent] No skill store available for setting skills.");
        return;
    }
    for (const auto & id : active_skills_) {
        context_manager_.remove_skill(id);
    }
    active_skills_.clear();


    for (const auto & id : skill_ids) {
        auto record = skill_store_->get_skill_record(id);
        if (record) {
            std::string content = skill_store_->get_skill_content(id);
            context_manager_.add_skill(id, content);
            active_skills_.insert(id);
            fprintf(stdout, "[Agent] Loaded skill: %s\n", record.value().name.c_str());
        }
        else {
            fprintf(stdout, "[Agent] Skill not found: %s\n", id.c_str());
        }
    }
}


void Agent::set_knowledge_directory(const std::string &path) {
    knowledge_base_ = std::make_shared<KnowledgeBase>(embedder_);
    if (!knowledge_base_->load_directory(path)) {
        fprintf(stderr, "[Agent] Knowledge directory load failed: %s\n", path.c_str());
        return;
    }
    tool_registry_.register_tool(std::make_shared<ListKnowledgeTool>(knowledge_base_));
    tool_registry_.register_tool(std::make_shared<SearchKnowledgeTool>(knowledge_base_));
    tool_registry_.register_tool(std::make_shared<LoadKnowledgeTool>(knowledge_base_));

    build_system_prompt();
    context_manager_.flush_pending();
}

void Agent::set_skill_directory(const std::string &path) {
    skill_store_ = std::make_shared<SkillStore>(embedder_);
    if (!skill_store_->load_directory(path)) {
        fprintf(stderr, "[Agent] Skill directory load failed: %s\n", path.c_str());
        return;
    }
    build_system_prompt();
    context_manager_.flush_pending();
}

void Agent::set_memory_file(const std::string &path) {
    memory_manager_ = std::make_shared<MemoryManager>(embedder_);
    if (!memory_manager_->load(path)) {
        fprintf(stderr, "[Agent] Memory manager could not load configuration: %s\n", path.c_str());
    }
    memory_path_ = path;
    tool_registry_.register_tool(std::make_shared<SearchMemoryTool>(memory_manager_));
    tool_registry_.register_tool(std::make_shared<RememberTool>(memory_manager_));
    build_system_prompt();
    context_manager_.flush_pending();
}


void Agent::save_memory() {
    if (memory_manager_ && !memory_path_.empty())
        memory_manager_->save(memory_path_);
}

std::string Agent::build_system_prompt() {
    std::ostringstream oss;
    if (!agent_role_description_.empty())
        oss << agent_role_description_;

    oss << "## Behaviour\n"
        << "Think through tasks step by step. When you need information, use tools — "
        << "do not guess, estimate, or fabricate values. Work autonomously until the task "
        << "is fully complete; do not stop to ask the user what to do next unless you are "
        << "genuinely blocked. Always produce a visible response — never end a turn with "
        << "only internal reasoning and no output.\n"
        << "Memory hints are partial. When they surface a topic (a company, event, or decision), "
        << "treat them as a starting point and use available tools to build a complete picture "
        << "before responding.\n";

    if (!agent_tools_description_.empty())
        oss << "\n\n" << agent_tools_description_;

    if (knowledge_base_ && knowledge_base_->size() > 0) {
        oss << "\n\n" << knowledge_base_->render_topic_tree();
        oss << "\nUse search_knowledge(query) whenever a topic arises that may have a relevant card. "
               "When card ids are already shown as hints, call load_knowledge(id) directly using "
               "those ids — do not invent or guess ids.\n";
    }
    if (memory_manager_) {
        oss << "\n\nYou have access to a persistent memory store of facts from past sessions. "
            << "Relevant facts are automatically surfaced as hints at the start of your turn, "
            << "but hints are partial — they reflect only the closest matches to the user's message. "
            << "Before using any tools, call search_memory(query) with specific terms to retrieve what is already known. "
            << "When a topic is raised and you want to recall more, call search_memory(query) "
            << "with specific terms to retrieve additional stored facts. "
            << "Use remember(fact) during a task to record important discoveries for future sessions.\n";
    }
    std::string system = oss.str();
    context_manager_.add_system(system);
    return system;
}


ContextStats Agent::context_stats() const { return context_manager_.get_stats(); }


AgentStats Agent::stats() const { return stats_; }


void Agent::begin_turn(TurnStats &ts) {
    ts.context_start = context_manager_.get_stats();
    context_manager_.reset_stats();
}

std::string Agent::build_enriched_user_message(const std::string &user_message) {
    std::string enriched = memory_enrich_user_message(user_message) + user_message;
    if (!knowledge_base_ || knowledge_base_->size() ==0) return enriched;

    auto hits = knowledge_base_->search(user_message, 5);
    if (hits.empty()) return enriched;

    std::string hints = "[Relevant knowledge - use load_knowledge(id) for full detail if needed]\n";
    for (const auto & h : hits) {
        hints += "- \"" + h.id + "\": " + h.summary + "\n";
        fprintf(stdout, "Knowledge Card Hint (%.2f): %s\n", h.score, h.title.c_str());
    }
    hints += "\n";
    return hints + enriched;
}


void Agent::run_generation_loop(StreamState &state, std::string &accumulated, std::string &final_response,
                                TurnStats &ts) {
    int context_full_count = 0;
    bool last_was_tool_call = false;
    int iter = 0;
    for (;iter < config_.tools.max_iterations; ++iter) {
        auto [gen, response] = run_generation_pass(state);
        ts.generation_passes.push_back({gen.tokens, gen.ms});
        final_response = response;

        if (gen.stop == GenerationStop::ContextFull) {
            if (!recover_from_context_full(state, accumulated, context_full_count, ts))
                break;
            continue;
        }
        context_full_count = 0;

        if (!state.has_complete_call()) { last_was_tool_call = false; break; }
        last_was_tool_call = true;
        if (!run_tool_calls(state, final_response, ts)) break;
    }

    if (iter == config_.tools.max_iterations && last_was_tool_call) {
        fprintf(stdout, "[Agent] Tool iteration limit (%d) reached — running final generation pass.\n",
                config_.tools.max_iterations);
        context_manager_.add_user(
            "[System: tool call limit reached for this turn — DO NOT CALL ANY TOOLS — SUMMARIZE YOUR FINDINGS AND WRITE A FINAL RESPONSE NOW]");
        auto [gen, response] = run_generation_pass(state);
        ts.generation_passes.push_back({ gen.tokens, gen.ms });
        if (!response.empty()) final_response = response;
    }
}


void Agent::finalize_turn(const std::string &user_message, const std::string &final_response, TurnStats &ts,
                          double turn_ms) {
    context_manager_.add_assistant(final_response);
    if (config_.knowledge.compaction_token_threshold > 0)
        context_manager_.compact_tool_results(config_.knowledge.compaction_token_threshold);

    ts.results_stubbed = context_manager_.stat_stubbed();
    ts.tick_events     = context_manager_.stat_ticks();
    ts.compressions    = context_manager_.stat_compressions();
    ts.total_ms        = turn_ms;
    ts.context_end     = context_manager_.get_stats();
    stats_.accumulate_turn_stats(ts);

    consolidate_memory(user_message, final_response);
}


bool Agent::init_rag_components() {
    if (!embedder_) {
        fprintf(stderr, "[Agent] No embedding service available for RAG services.\n");
        return false;
    }

    if (!config_.skills.directory_path.empty() && !config_.skills.skill_ids_to_set.empty()) {
        std::string dir_path = ".";
        if (Utility::validate_dir(config_.skills.directory_path)) {
            dir_path = config_.skills.directory_path;
            fprintf(stdout, "[Agent] Skills directory: %s\n", dir_path.c_str());
        } else {
            fprintf(stdout,"[Agent] Could not validate skills directory, using '.' working directory.\n");
        }
        set_skill_directory(dir_path);
        set_skills(config_.skills.skill_ids_to_set);
    }

    if (!config_.knowledge.directory_path.empty()) {

        std::string dir_path = ".";
        if (Utility::validate_dir(config_.knowledge.directory_path)) {
            dir_path = config_.knowledge.directory_path;
            fprintf(stdout, "[Agent] Knowledge directory: %s\n", dir_path.c_str());
        } else {
            fprintf(stdout,"[Agent] Could not validate knowledge directory, using '.' working directory.\n");
        }
        set_knowledge_directory(dir_path);
    }

    if (!config_.memory.memory_file_path.empty()) {
        set_memory_file(config_.memory.memory_file_path);
    }

    return true;
}


void Agent::consolidate_memory(const std::string &user_message, const std::string &assistant_response) {
    if (!memory_manager_ || assistant_response.size() < config_.memory.consolidate_assistant_threshold) return;

    fprintf(stdout, "[Agent] Extracting facts from turn to give memory hints...");
    auto facts = summarizer_->extract_facts(user_message, assistant_response);
    if (facts.size() > 0) std::cout << "[Agent] Facts extracted:\n";
    for (const auto & fact : facts) {
        auto hit = memory_manager_->similar_exists(fact);
        if (hit) {
            fprintf(stdout, "- Similar Fact:\n  Candidate: %s\n  Match (%.2f): %s\n\n", fact.c_str(), hit->score, hit->data.c_str());
            continue;
        }
        fprintf(stdout, "- %s\n", fact.c_str());
        memory_manager_->insert(fact);
    }
}


std::string Agent::build_memory_hints(const std::string &query) const {
    if (!memory_manager_ || memory_manager_->size() == 0) return {};

    const std::string key = query.size() > 512 ? query.substr(0, 512) : query;
    auto hits = memory_manager_->search(key, 2, config_.memory.hint_min_score);
    if (hits.empty()) return {};
    std::string block = "\n[Memory hints]\n";
    for (auto & h : hits) {
        block += "- " + h.data + "\n";
    }
    block += "Call search_memory(query) to retrieve more stored facts on this topic.\n";
    fprintf(stdout, "[Agent] Memory Hint Given:\n%s\n", block.c_str());
    return block;
}


std::string Agent::memory_enrich_user_message(const std::string &user_message) {
    if (memory_manager_ && memory_manager_->size() > 0) {
        std::string key = summarizer_ ? summarizer_->extract_query(user_message) : user_message;
        if (key.empty()) key = user_message;
        auto hits = memory_manager_->search(key, config_.memory.hint_top_k, config_.memory.hint_min_score);
        if (!hits.empty()) {
            std::string block = "[Memory — recalled facts]\n";
            for (auto & h : hits)
                block += "- " + h.data + "\n";
            block += "Call search_memory(query) to retrieve more stored facts on this topic.\n";
            fprintf(stdout, "[Agent]%s", block.c_str());
            return block;
        }
    }
    return {};
}


GenerationPass Agent::run_generation_pass(StreamState &state) {
    state.reset_for_next_pass();
    auto & cb = config_.callbacks;
    auto streaming_cb = [&](const std::string & token) {
        return stream_processor_->process(token, state);
    };
    auto result = prompt_->generate_continue(*context_, *sampler_, streaming_cb);
    if (!state.pending_tag.empty()) {
        if (cb.on_token) cb.on_token(state.pending_tag);
        state.visible_response += state.pending_tag;
        state.pending_tag.clear();
    }
    std::string response = state.visible_response;
    auto first = response.find_first_not_of(" \n\r\t");
    if (first != std::string::npos && first > 0) response = response.substr(first);
    return { result, response };
}


bool Agent::run_tool_calls(const StreamState &state, const std::string & final_response, TurnStats &stats) {
    auto & cb = config_.callbacks;
    if (!final_response.empty()) context_manager_.add_assistant(final_response);
    for (const auto & call : state.completed_tool_calls)
        context_manager_.add_tool_call(call);
    const auto & calls = state.completed_tool_calls;
    for (size_t i = 0; i < calls.size(); ++i) {
        const auto & call = calls[i];
        ToolCallStats tcs;
        tcs.name = call.name;
        ToolResult result;
        if (call.name == "__malformed__") {
            result.success = false;
            result.error = "Malformed call. Raw: " + call.arguments.value("raw", "")
             + " Error: " + call.arguments.value("error", "unknown");
        } else {
            Stopwatch sw;
            result = tool_registry_.execute_tool(call.name, call.arguments);
            tcs.exec_ms = sw.ms();
            if (result.success && (int)result.result.size() > config_.memory.midturn_result_min_len) {
                std::string hints = build_memory_hints(result.result);
                if (!hints.empty()) result.result += hints;
            }
        }
        tcs.success = result.success;
        if (cb.on_tool_result) cb.on_tool_result(call, result);
        Stopwatch encode_sw;
        bool ok = context_manager_.add_tool_result(call, result ,i == calls.size() - 1);
        tcs.encode_ms = encode_sw.ms();
        stats.tool_calls.push_back(tcs);
        if (!ok) {
            fprintf(stderr, "[Agent] Failed to encode tool result '%s'.\n", call.name.c_str());
            return false;
        }
    }
    sampler_->reset();
    return true;

}


bool Agent::recover_from_context_full(StreamState &state, std::string &accumulated, int &count, TurnStats &stats) {
    if (++count > 2) {
        fprintf(stderr, "[Agent] Repeated context full - aborting turn.\n");
        return false;
    }
    ++stats.context_full_events;
    const std::string & partial = state.visible_response;
    if (!partial.empty()) {
        accumulated += partial;
        state.has_complete_call()
        ? context_manager_.add_assistant(partial) : context_manager_.add_assistant_partial(partial);
    }
    for (const auto & call : state.completed_tool_calls)
        context_manager_.add_tool_call(call);
    context_manager_.compact_tool_results(config_.knowledge.compaction_token_threshold);
    context_manager_.tick();
    Stopwatch sw;
    context_manager_.flush_pending();
    stats.flush_ms += sw.ms();
    sampler_->reset();
    return true;
}


void Agent::clear_skills() {
    context_manager_.clear_skills();
}
