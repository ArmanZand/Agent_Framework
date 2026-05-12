#ifndef AGENT_FRAMEWORK_MEMORYMANAGER_H
#define AGENT_FRAMEWORK_MEMORYMANAGER_H
#include <algorithm>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>

#include "../../infrastructure/BM25.h"
#include "../../infrastructure/Utility.h"
#include "../../model_services/EmbeddingService.h"

#include "nlohmann/json.hpp"
using json = nlohmann::ordered_json;
struct MemoryHit {
    std::string id;
    std::string data;
    float score = 0.f;
};

struct MemoryItem {
    std::string id;
    std::string data;
    int insertion_order = 0;
    std::vector<float> embeddings;
    struct {
        int data_len = 0;
        std::unordered_map<std::string, int> tf_data;
        std::unordered_set<std::string> unique_terms;
    } bm25;
    json to_json() const;
    static MemoryItem from_json(const json & j);
};

class MemoryManager {
public:
    MemoryManager(std::shared_ptr<EmbeddingService> embedder);
    bool insert(const std::string & data);
    std::vector<MemoryHit> search(const std::string & query, int top_k = 5, float min_score = 0.7f, float recency_weight = 0.f) const;
    std::optional<MemoryHit> similar_exists(const std::string &query, float min_score = 0.9f) const;

    bool save(const std::string & path);
    bool load(const std::string & path);
    int size() const;

private:
    float bm25_score(const MemoryItem & item, const std::vector<std::string> & tokens, const std::unordered_map<std::string, float> & idf) const;
    std::vector<MemoryHit> bm25_query(const std::vector<std::string> &tokens,
    const std::unordered_map<std::string, float> &idf, int top_k = 5, float min_score = 0.3f) const;
    std::vector<MemoryHit> embed_query(const std::string & query, int top_k = 5, float min_score = 0.7f) const;
    static std::vector<std::string> deduplicated_tokens(const std::string & text);
    std::unordered_map<std::string, float> build_idf_cache(const std::vector<std::string> & tokens) const;
    void increase_avg_len(size_t new_len);

    std::shared_ptr<EmbeddingService> embedder_;
    std::unordered_map<std::string, MemoryItem> items_;
    std::unordered_map<std::string, int> doc_freq_;

    bool dirty_ = false;
    float avg_len_ = 0;
    int item_count_ = 0;

    struct {
        float bm25_weight_ = 0.4f;
        float embed_weight_ = 0.6f;
        float bm25_min_score_ = 0.4f;
        float embed_min_score_ = 0.7f;
    } search_norm_param;
};


#endif //AGENT_FRAMEWORK_MEMORYMANAGER_H