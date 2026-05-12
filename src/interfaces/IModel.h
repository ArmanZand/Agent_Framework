#ifndef AGENT_FRAMEWORK_IMODEL_H
#define AGENT_FRAMEWORK_IMODEL_H

#include <functional>
#include <string>
#include <memory>
#include <cstdint>

class IModel {
public:
    virtual ~IModel() = default;

    struct GenerateParams {
        std::string prompt;
        float temperature;
        float top_p;
        float min_p;
        int32_t top_k;
        int max_tokens = -1;
        std::function<void(const std::string &)> on_token = nullptr;
    };

    struct GenerateResult {
        std::string text;
        int tokens_generated = 0;
        bool success = false;
        std::string error;
    };

    struct Message {
        std::string role;
        std::string content;
    };

    virtual GenerateResult generate(const GenerateParams& params) = 0;

    struct Session {
        virtual ~Session() = default;
    };
    virtual std::unique_ptr<Session> create_session(int n_ctx = 2048) = 0;
    virtual GenerateResult generate_with_session(Session * session, const GenerateParams & params) = 0;

    virtual int get_context_size() const = 0;
    virtual std::string get_model_name() const = 0;
};

struct ModelConfig {
    std::string model_path;
    std::string model_name;
    int n_gpu_layers = 99;
    int n_ctx = 2048;
    int n_batch = 2048;
    int n_threads = 1;
    int n_threads_batch = 1;
    bool verbose = false;
};

#endif //AGENT_FRAMEWORK_IMODEL_H