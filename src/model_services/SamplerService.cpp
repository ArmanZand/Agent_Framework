#include "SamplerService.h"

#include "BackendService.h"

SamplerService::SamplerService(const SamplerConfig &config) : config_(config) {
}

SamplerService::~SamplerService() {
    if (sampler_ && BackendService::instance().is_running()) {
        BackendService::instance().llama().sampler_free(sampler_);
        if (config_.verbose) {
            fprintf(stdout, "[SamplerService] Unloaded sampler.\n");
        }
    }
}

llama_token SamplerService::sample(llama_context * context, int index) {
    if (!sampler_ && !context) {
        return llama_token{};
    }
    return BackendService::instance().llama().sampler_sample(sampler_, context, index);
}

void SamplerService::set_grammar(llama_vocab * vocab, const std::string &gbnf) {
    auto & llama = BackendService::instance().llama();
    sampler_ = llama.sampler_init_grammar(
        vocab, gbnf.c_str(), "root");
    rebuild_chain();
}

void SamplerService::use_greedy() {
    config_.top_k = 1;
    config_.top_p = 1.0f;
    config_.min_p = 0.0f;
    config_.temperature = 1.0f;
    rebuild_chain();
}

void SamplerService::use_creative() {
    config_.top_k = 40;
    config_.top_p = 0.95f;
    config_.min_p = 0.0f;
    config_.temperature = 0.8f;
    rebuild_chain();
}

void SamplerService::use_exploration() {
    config_.top_k = 0;
    config_.top_p = 1.0f;
    config_.min_p = 0.05f;
    config_.temperature = 1.2f;
    rebuild_chain();
}

void SamplerService::set_temperature(float temperature) {
    config_.temperature = temperature;
    rebuild_chain();
}

bool SamplerService::initialize() {
    auto & backend = BackendService::instance();
    if (!backend.is_running()) {
        if (!backend.start())
            return false;
    }

    if (!load_sampler())
        return false;

    rebuild_chain();

    return true;
}

void SamplerService::reset() const {
    if (sampler_) {
        auto & llama = BackendService::instance().llama();
        llama.sampler_reset(sampler_);
    }
}

bool SamplerService::load_sampler() {
    auto & llama = BackendService::instance().llama();
    if (config_.verbose) {
        fprintf(stdout, "[SamplerService] Loading sampler - ");
        fflush(stdout);
    }
    sampler_ = llama.sampler_chain_init(llama.sampler_chain_default_params());
    if (!sampler_) {
        fprintf(stderr, "[SamplerService] Failed to load sampler\n");
        return false;
    }
    if (config_.verbose) {
        fprintf(stdout, "DONE\n");
        fflush(stdout);
    }
    return true;
}

void SamplerService::rebuild_chain() {
    auto & llama = BackendService::instance().llama();
    if (sampler_) {
        llama.sampler_free(sampler_);
        sampler_ = llama.sampler_chain_init(llama.sampler_chain_default_params());
    }
    if (config_.top_k > 0)
        llama.sampler_chain_add(sampler_, llama.sampler_init_top_k(config_.top_k));
    if (config_.top_p < 1.0f)
        llama.sampler_chain_add(sampler_, llama.sampler_init_top_p(config_.top_p, 1));
    if (config_.min_p > 0.0f)
        llama.sampler_chain_add(sampler_, llama.sampler_init_min_p(config_.min_p, 1));

    llama.sampler_chain_add(sampler_, llama.sampler_init_temp(config_.temperature));

    int seed = (config_.seed == 0) ? LLAMA_DEFAULT_SEED : config_.seed;
    llama.sampler_chain_add(sampler_, llama.sampler_init_dist(seed));

}
