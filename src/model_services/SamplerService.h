#ifndef AGENT_FRAMEWORK_SAMPLERSERVICE_H
#define AGENT_FRAMEWORK_SAMPLERSERVICE_H
#include <string>

#include "llama.h"

struct SamplerConfig {
    float temperature = 0.7f;
    float top_p = 0.8f;
    float min_p = 0.0f;
    int top_k = 20;
    int seed = 0;
    bool verbose = false;
};

class SamplerService {
public:
    SamplerService(const SamplerConfig & config);
    ~SamplerService();

    SamplerService(const SamplerService &) = delete;
    SamplerService & operator=(const SamplerService &) = delete;

    bool initialize();
    void reset() const;
    llama_token sample(llama_context * context, int index = -1);
    void set_grammar(llama_vocab * vocab, const std::string & gbnf);

    void use_greedy();
    void use_creative();
    void use_exploration();

    void set_temperature(float temperature);

    llama_sampler * raw() const { return sampler_; }
private:
    bool load_sampler();
    void rebuild_chain();

    SamplerConfig config_;
    llama_sampler * sampler_ = nullptr;;
};


#endif //AGENT_FRAMEWORK_SAMPLERSERVICE_H