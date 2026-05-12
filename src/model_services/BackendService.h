#ifndef AGENT_FRAMEWORK_BACKENDSERVICE_H
#define AGENT_FRAMEWORK_BACKENDSERVICE_H
#include "../models/LlamaWrapper.h"

#include <optional>
#include <string>

class BackendService {
public:
    ~BackendService();

    BackendService(const BackendService&) = delete;
    BackendService(BackendService&&) = delete;
    BackendService operator=(const BackendService&) = delete;
    BackendService operator=(BackendService&&) = delete;

    static BackendService & instance() {
        static BackendService singleton{};
        return singleton;
    }
    bool start();
    void stop();
    bool is_running() { return is_running_; }
    LlamaWrapper & llama() { return llama_; }
private:
    bool is_running_ = false;
    LlamaWrapper llama_;

    BackendService();


};


#endif //AGENT_FRAMEWORK_BACKENDSERVICE_H