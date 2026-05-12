#include "EmbeddingService.h"

#include <algorithm>
#include <cmath>

#include "BackendService.h"

EmbeddingService::EmbeddingService(std::shared_ptr<PromptService> prompt_service,
    std::shared_ptr<ContextService> context_service, const EmbeddingConfig &config) : prompt_service_(std::move(prompt_service)),
context_service_(std::move(context_service)), config_(config) {
}

std::shared_ptr<EmbeddingService> EmbeddingService::create(EmbeddingConfig &config) {
    BackendService::instance().start();
    config.context.generate_embeddings = true;

    auto model = ModelService::create(config.model);
    if (!model) {
        fprintf(stderr, "[EmbeddingService] Failed to initialize model '%s'.\n", config.model.model_path.c_str());
        return nullptr;
    }

    auto context = std::make_shared<ContextService>(model, config.context);
    if (!context->initialize()) {
        fprintf(stderr, "[EmbeddingService] Failed to initialize context.\n");
        return nullptr;
    }


    auto prompt = std::make_shared<PromptService>(model, config.prompt);
    return std::make_shared<EmbeddingService>(prompt, context, config);
}

std::vector<float> EmbeddingService::embed(const std::string &text) {
    auto vec = prompt_service_->generate_embedding(*context_service_, text);
    if (config_.normalize && !vec.empty())
        l2_normalize(vec);
    return vec;
}

std::vector<std::vector<float>> EmbeddingService::embed_batch(const std::vector<std::string> &texts) {
    std::vector<std::vector<float>> results;
    results.reserve(texts.size());
    for (const auto & t : texts) {
        auto v = embed(t);
        results.push_back(std::move(v));
    }
    return results;
}

float EmbeddingService::cosine_similarity(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return denom > 1e-8f ? dot / denom : 0.0f;
}

std::vector<std::pair<float, size_t>> EmbeddingService::search_scores(std::span<const float> & query_embedding, std::span<const std::span<const float>> embeddings, int top_k) {
    std::vector<std::pair<float, size_t>> scores;
    for (size_t i = 0; i < embeddings.size(); ++i) {
        float score = cosine_similarity(query_embedding, embeddings[i]);
        scores.push_back({score, i});
    }
    int k = (std::min)(top_k, static_cast<int>(scores.size()));
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
        [](const auto & a, const auto & b) {
        return a.first > b.first;
    });
    return scores;
}

void EmbeddingService::l2_normalize(std::vector<float> &vec) {
    float norm = 0.0f;
    for (float v : vec) norm += v * v;
    norm = std::sqrt(norm);
    if (norm < 1e-8f) return;
    for (float & v : vec) v /= norm;
}
