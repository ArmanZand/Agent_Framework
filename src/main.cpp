#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "agents/Agent.h"
#include "model_services/BackendService.h"
#include "model_services/EmbeddingService.h"

#include "infrastructure/Utility.h"

#include "tools/io/ExploreDirectoryTool.h"
#include "tools/io/FileReadTool.h"
#include "tools/io/FileInfoTool.h"

struct AppConfig {
    std::string model_path;
    std::string embed_path;
    std::string embedder_id;
    std::string knowledge_dir;
    std::string skills_dir;
    std::string workspace_dir;
    std::string memory_file;
    std::vector<std::string> skill_ids;
    int context_size = 32000;
    int batch_size = 1000;
    static AppConfig  from_json(const json & j) {
        AppConfig c;
        c.model_path = j.value("model_path", "");
        c.embed_path = j.value("embed_path", "");
        c.embedder_id = j.value("embedder_id", "");
        c.knowledge_dir = j.value("knowledge_dir", "data/knowledge");
        c.skills_dir = j.value("skills_dir", "data/skills");
        c.workspace_dir = j.value("workspace_dir", "data/workspace");
        c.memory_file = j.value("memory_file", "data/memory.json");
        c.skill_ids = j.value("skill_ids", std::vector<std::string>());
        c.context_size = j.value("context_size", 32000);
        c.batch_size = j.value("batch_size", 1000);
        return c;
    }
};

AppConfig load_config(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "Cannot open config: %s\n", path.c_str());
        std::exit(1);
    }
    json j; f >> j;
    return AppConfig::from_json(j);
}


bool on_token(const std::string & token) {
    std::cout << token << std::flush;
    return true;
}

bool on_thinking(const std::string & token) {
    std::cout << "\033[33m" << token << "\033[0m" << std::flush;
    return true;
}

void on_tool_start(const ToolCall & tool_call) {
    std::cout << "\033[35m";
    std::cout << "Tool Called - " << tool_call.name << "(" << tool_call.arguments << ")";
    std::cout << "\033[0m" << std::endl << std::flush;
}

void on_tool_result(const ToolCall & tool_call, const ToolResult & result) {

    std::cout << "\033[35m";
    std::cout << "Tool Result - " << tool_call.name << "(" << tool_call.arguments << ") ";
    std::cout << "\033[0m";
    if (result.success) {
        std::cout << "\033[1m\033[32m";
        std::cout << "✓ SUCCESS\n";
    } else {
        std::cout << "\033[1m\033[31m";
        std::cout << "✗ FAIL\n";
    }
    std::cout << "\033[0m" << std::endl << std::flush;
}

void on_thinking_start() {
    std::cout << "\033[33m" << "> THINKING START" << "\033[0m" << std::flush;
}
void on_think_end() {
    std::cout << std::endl << std::flush;
}

AgentConfig create_agent_config(const AppConfig & app_config) {
    AgentConfig config;
    config.context.context_size = app_config.context_size;
    config.context.batch_size = app_config.batch_size;
    config.callbacks.on_token = on_token;
    config.callbacks.on_thinking_token = on_thinking;
    config.callbacks.on_thinking_start = nullptr;
    config.callbacks.on_thinking_end = on_think_end;
    config.callbacks.on_tool_start = on_tool_start;
    config.callbacks.on_tool_result = on_tool_result;

    config.skills.directory_path = app_config.skills_dir;
    config.skills.skill_ids_to_set = app_config.skill_ids;
    config.knowledge.directory_path = app_config.knowledge_dir;
    config.memory.memory_file_path = app_config.memory_file;
    return config;
}

std::shared_ptr<EmbeddingService> create_embedding_service(const AppConfig & app_config) {
    EmbeddingConfig config;
    config.model.model_path = app_config.embed_path;
    config.context.context_size = 512;
    config.prompt.get_default_template = false;
    config.id = app_config.embedder_id;
    return EmbeddingService::create(config);
}


std::vector<std::shared_ptr<ITool>> create_tool_list(const AppConfig & app_config) {
    std::vector<std::shared_ptr<ITool>> tools;
    if (Utility::validate_dir(app_config.workspace_dir)) {
        tools.push_back(std::make_shared<ExploreDirectoryTool>(app_config.workspace_dir));
        tools.push_back(std::make_shared<FileReadTool>(app_config.workspace_dir));
        tools.push_back(std::make_shared<FileInfoTool>(app_config.workspace_dir));
        fprintf(stdout, "Valid directory for agent workspace, created I/O tool list.");
    } else {
        fprintf(stdout, "Invalid directory for agent workspace, omitting I/O tools.");
    }
    return tools;
}


int main(int argc, char ** argv) {
    std::string config_path = argc > 1 ? argv[1] : "config.json";
    AppConfig cfg = load_config(config_path);

    BackendService::instance().start();
    AgentConfig agent_config = create_agent_config(cfg);
    ModelConfig model_config = { .model_path = cfg.model_path, .gpu_layers = 99 };
    auto embedder = create_embedding_service(cfg);

    auto model = ModelService::create(model_config);

    auto agent = Agent::create(agent_config, model, embedder);
    if (!agent) {
        std::cout << agent_config.name << " failed to be created." << std::endl;
        return 1;
    }
    agent->set_tools(create_tool_list(cfg));
    agent->build_system_prompt();
    while (true) {
        agent->context_stats().print();
        printf("\033[32m> \033[0m");
        std::string question;
        std::getline(std::cin, question);
        if (question.empty())
            break;
        agent->chat(question);
        agent->stats().print_turn();
    }
    agent->stats().print_cumulative();

    return 0;
}