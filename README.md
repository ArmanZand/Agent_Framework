# Agent Framework
This is a work-in-progress snapshot of an agent framework implementation in C++ with [llama.cpp](https://github.com/ggml-org/llama.cpp) on windows.
For now it is a single-agent, local-only with token streaming to console. Tested with CUDA-enabled llama.cpp builds.
The framework can be used to enable multi-agent decisions as agents can be setup with unique parameters and tools. 

## Components
The agent has several components to improve the quality of its responses:
- Tool Calling - The model produces tags that contain JSON for tool calling.
- Skill Store - Inserts skills defined in Markdown (.md) files into the context.
- Knowledge-Base - Enriches the model context by loading 'knowledge cards' from the knowledge-base when there is relevance.
- Memory Manager - Extracts atomic facts from the conversation turn to remind the agent of previous information.
- Data Hints - Injects a short hint of relevant knowledge cards or facts on user query or tool calls.



## How to run
### 1) Obtain the pre-built dlls
From the [llama.cpp](https://github.com/ggml-org/llama.cpp) project download windows x64 CUDA 13 DLLs. 
In the project directory create a folder 'dll' so that these dll files exist:
```
dll/llama.dll
dll/ggml.dll
dll/ggml-base.dll
dll/ggml-cpu-x64.dll
dll/ggml-cuda.dll
dll/cudart64_13.dll
dll/cublas64_13.dll
dll/cublasLt64_13.dll
```
CMake is configured to pull these into the compiled output directory.
### 2) Obtain models
Explore https://huggingface.co for llama.cpp and hardware compatible models.

This project has been run with the [Qwen3.5-9B Q4_K_M](https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF)
model for generating responses / summarization. For embeddings a [Gemma-300M-Q8_0](https://huggingface.co/Bhuvneesh/embeddinggemma-300m-Q8_0-GGUF) model.

### 3) Build and Run
Requires:
- CMake >= 3.20
- C++20 compiler
- llama.cpp 
- nlohmann/json

```bash
cmake -B build
cmake --build build
```

Complete a configuration file by copying `config.example.json` and editing the fields.

Run:

`agent_framework.exe config.json`


## License

MIT — see [LICENSE](LICENSE).