#ifndef AGENT_FRAMEWORK_EMBEDDINGSERVICE_H
#define AGENT_FRAMEWORK_EMBEDDINGSERVICE_H
#include <span>

#include "PromptService.h"

struct EmbeddingConfig {
    bool normalize = true;
    bool verbose = false;

    ModelConfig model;
    ContextConfig context;
    PromptConfig prompt;
    std::string id = "unknown_embedder";
};
class EmbeddingService {
public:
    EmbeddingService(std::shared_ptr<PromptService> prompt_service, std::shared_ptr<ContextService> context_service, const EmbeddingConfig & config);
    static std::shared_ptr<EmbeddingService> create(EmbeddingConfig & config);

    std::vector<float> embed(const std::string & text);
    std::vector<std::vector<float>> embed_batch(const std::vector<std::string> & texts);

    std::string id() const { return config_.id;}

    static float cosine_similarity(std::span<const float> a, std::span<const float> b);

    static std::vector<std::pair<float, size_t>> search_scores(std::span<const float> & query_embedding, std::span<const std::span<const float>> embeddings, int top_k);
private:
    void l2_normalize(std::vector<float> & vec);

    std::shared_ptr<PromptService> prompt_service_;
    std::shared_ptr<ContextService> context_service_;
    EmbeddingConfig config_;
};


#endif //AGENT_FRAMEWORK_EMBEDDINGSERVICE_H