cmake_minimum_required(VERSION 3.20)
project(agent_framework)

set(CMAKE_CXX_STANDARD 20)

add_executable(agent_framework src/main.cpp
        src/models/LlamaWrapper.cpp
        src/models/LlamaWrapper.h
        src/models/LlamaModel.cpp
        src/models/LlamaModel.h
        src/tools/InternalTools.h
        src/models/EmbeddingModel.cpp
        src/models/EmbeddingModel.h
        src/interfaces/IToolRegistry.h
        src/interfaces/ITool.h
        src/tools/ToolRegistry.cpp
        src/tools/ToolRegistry.h
        src/infrastructure/MemoryConfig.h
        src/infrastructure/SamplingConfig.h
        src/infrastructure/ToolConfig.h
        src/model_services/BackendService.cpp
        src/model_services/BackendService.h
        src/model_services/ModelService.cpp
        src/model_services/ModelService.h
        src/model_services/ContextService.cpp
        src/model_services/ContextService.h
        src/model_services/SamplerService.cpp
        src/model_services/SamplerService.h
        src/model_services/PromptService.cpp
        src/model_services/PromptService.h
        src/model_services/SummarizationService.cpp
        src/model_services/SummarizationService.h
        src/model_services/EmbeddingService.cpp
        src/model_services/EmbeddingService.h
        src/agents/Agent.h
        src/agents/Agent.cpp
        src/rag/skills/SkillStore.cpp
        src/rag/skills/SkillStore.h
        src/infrastructure/ContextManager.cpp
        src/infrastructure/ContextManager.h
        src/infrastructure/ChatMessage.h
        src/rag/knowledge/Card.h
        src/rag/knowledge/KnowledgeBase.cpp
        src/rag/knowledge/KnowledgeBase.h
        src/tools/knowledge/ListKnowledgeTool.h
        src/tools/knowledge/ListKnowledgeTool.cpp
        src/tools/knowledge/SearchKnowledgeTool.cpp
        src/tools/knowledge/SearchKnowledgeTool.h
        src/infrastructure/AgentStats.h
        src/infrastructure/Stopwatch.h
        src/infrastructure/BM25.cpp
        src/infrastructure/BM25.h
        src/rag/memory/MemoryManager.cpp
        src/rag/memory/MemoryManager.h
        src/infrastructure/Utility.h
        src/infrastructure/Keccak.cpp
        src/infrastructure/Keccak.h
        src/tools/memory/SearchMemoryTool.cpp
        src/tools/memory/SearchMemoryTool.h
        src/tools/memory/RememberTool.cpp
        src/tools/memory/RememberTool.h
        src/tools/io/ExploreDirectoryTool.cpp
        src/tools/io/ExploreDirectoryTool.h
        src/tools/io/FileInfoTool.cpp
        src/tools/io/FileInfoTool.h
        src/tools/io/FileReadTool.cpp
        src/tools/io/FileReadTool.h
        src/tools/io/FileWriteTool.cpp
        src/tools/io/FileWriteTool.h
        src/tools/knowledge/LoadKnowledgeTool.cpp
        src/agents/StreamProcessor.cpp
        src/agents/StreamProcessor.h
)

target_include_directories(agent_framework PRIVATE ${CMAKE_SOURCE_DIR}/include)

if (WIN32)
    add_custom_command(TARGET agent_framework POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/dll/llama.dll"
        "${CMAKE_SOURCE_DIR}/dll/ggml.dll"
        "${CMAKE_SOURCE_DIR}/dll/ggml-base.dll"
        "${CMAKE_SOURCE_DIR}/dll/ggml-cpu-x64.dll"
        "${CMAKE_SOURCE_DIR}/dll/ggml-cuda.dll"
        "${CMAKE_SOURCE_DIR}/dll/cudart64_13.dll"
        "${CMAKE_SOURCE_DIR}/dll/cublas64_13.dll"
        "${CMAKE_SOURCE_DIR}/dll/cublasLt64_13.dll"
        $<TARGET_FILE_DIR:agent_framework>)
endif()