#include "MemoryManager.h"

json MemoryItem::to_json() const {
    json j;
    j["id"] = id;
    j["data"] = data;
    j["insertion_order"] = insertion_order;
    j["embeddings"] = embeddings;
    j["bm25"] = {
        {"data_len", bm25.data_len },
        {"tf_data", bm25.tf_data},
        {"unique_terms", bm25.unique_terms}
    };
    return j;
}

MemoryItem MemoryItem::from_json(const json &j) {
    MemoryItem item;
    item.id = j.value("id", "");
    item.data = j.value("data", "");
    item.insertion_order = j.value("insertion_order", 0);
    if (j.contains("embeddings") && j["embeddings"].is_array())
        item.embeddings = j["embeddings"].get<std::vector<float>>();

    if (j.contains("bm25")) {
        const auto & b = j["bm25"];
        item.bm25.data_len = b.value("data_len", 0);
        if (b.contains("tf_data"))
            item.bm25.tf_data = b["tf_data"].get<std::unordered_map<std::string, int>>();
        if (b.contains("unique_terms"))
            for (const auto & t : b["unique_terms"])
                item.bm25.unique_terms.insert(t.get<std::string>());
    }
    return item;
}

MemoryManager::MemoryManager(std::shared_ptr<EmbeddingService> embedder): embedder_(std::move(embedder)) {}

bool MemoryManager::insert(const std::string &data) {
    if (!embedder_ || data.empty()) return false;
    const std::string id = Utility::compute_hash(data);
    if (items_.contains(id)) return false;

    auto embedding = embedder_->embed(data);
    if (embedding.empty()) return false;

    MemoryItem item;
    item.id = id;
    item.data = data;
    item.embeddings = std::move(embedding);
    item.insertion_order = item_count_;
    BM25::extract_BM25_stats(item.data, item.bm25.data_len, item.bm25.tf_data, item.bm25.unique_terms);
    for (const auto & t : item.bm25.unique_terms) doc_freq_[t]++;
    increase_avg_len(item.bm25.data_len);
    items_.emplace(id, std::move(item));
    dirty_ = true;
    return true;
}

std::vector<MemoryHit> MemoryManager::search(const std::string &query, int top_k, float min_score, float recency_weight) const {
    if (!embedder_ || query.empty()) return {};

    auto tokens = deduplicated_tokens(query);
    auto idf = build_idf_cache(tokens);

    const int candidate_k = top_k * 3;
    auto bm25_hits = bm25_query(tokens, idf, candidate_k, search_norm_param.bm25_min_score_);
    auto embed_hits = embed_query(query, candidate_k, search_norm_param.embed_min_score_);

    std::unordered_map<std::string, float> rrf;
    rrf.reserve(bm25_hits.size() + embed_hits.size());
    auto fuse = [&](const std::vector<MemoryHit> & hits, float weight) {
        for (size_t i = 0; i < hits.size();++i)
            rrf[hits[i].id] += weight / (60.0f + static_cast<float>(i + 1));
    };
    fuse(bm25_hits, search_norm_param.bm25_weight_);
    fuse(embed_hits, search_norm_param.embed_weight_);

    

    if (recency_weight > 0.f && item_count_ > 1) {
        for (auto & [id, score] : rrf) {
            auto it = items_.find(id);
            if (it == items_.end()) continue;
            float recency = static_cast<float>(it->second.insertion_order) / static_cast<float>(item_count_ - 1);
            score += recency_weight * recency;
        }
    }


    std::vector<std::pair<std::string, float>> ranked(rrf.begin(), rrf.end());
    std::sort(ranked.begin(), ranked.end(),
        [](const auto & a, const auto & b) { return a.second > b.second; });


    if (ranked.empty() || ranked.front().second <= 0.f) return {};
    const float inv = 1.f / ranked.front().second;
    for (auto & [id, s] : ranked) s *= inv;



    std::vector<MemoryHit> out;
    out.reserve(top_k);
    for (auto & [id, score] : ranked) {
        if (score < min_score) break;
        auto it = items_.find(id);
        if (it == items_.end()) continue;
        out.push_back({id, it->second.data, score});
        if (static_cast<int>(out.size()) >= top_k) break;
    }
    return out;
}

std::optional<MemoryHit> MemoryManager::similar_exists(const std::string &query, float min_score) const {
    if (!embedder_ || query.empty()) return std::nullopt;
    auto hits = embed_query(query, 1, min_score);
    return (!hits.empty()) ? std::optional<MemoryHit>(hits.front()) : std::nullopt;
}

bool MemoryManager::save(const std::string &path) {
    if (!dirty_) {
        return true;
    }
    json j;
    j["item_count"] = item_count_;
    j["avg_len"] = avg_len_;
    j["doc_freq"] = doc_freq_;
    j["items"] = json::array();
    j["embedder_model"] = embedder_->id();

    for (const auto & [id, item] : items_) {
        j["items"].push_back(std::move(item.to_json()));
    }
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "[MemoryManager] cannot write to %s\n", path.c_str());
        return false;
    }
    f << j.dump(2);
    fprintf(stdout, "[MemoryManager] Saved %zu items to %s\n", items_.size(), path.c_str());
    dirty_ = false;
    return true;
}

bool MemoryManager::load(const std::string &path) {
    std::ifstream f(path);
    if (!f) return false;

    json j;
    try { f >> j; }
    catch (...) {
        fprintf(stderr, "[MemoryManager] Failed to parse %s\n", path.c_str());
        return false;
    }

    std::string embedding_model = j.value("embedder_model", "");
    if (embedding_model != embedder_->id()) {
        fprintf(stdout, "[MemoryManager] embedding model '%s' != current '%s', ignoring cache\n",
            embedding_model.c_str(), embedder_->id().c_str());
        return false;
    }

    items_.clear();
    doc_freq_.clear();
    item_count_ = j.value("item_count", 0);
    avg_len_ = j.value("avg_len", 0.f);
    if (j.contains("doc_freq"))
        doc_freq_ = j["doc_freq"].get<std::unordered_map<std::string, int>>();


    for (const auto & e : j.value("items", json::array())) {
        MemoryItem item = MemoryItem::from_json(e);
        if (item.id.empty() || item.data.empty()) continue;
        items_.emplace(item.id, std::move(item));
    }
    fprintf(stdout, "[MemoryManager] Loaded %zu items from %s\n", items_.size(), path.c_str());
    return true;
}

int MemoryManager::size() const { return static_cast<int>(items_.size()); }

float MemoryManager::bm25_score(const MemoryItem &item, const std::vector<std::string> &tokens,
                                const std::unordered_map<std::string, float> &idf) const {
    float score = 0.f;
    for (const auto & q : tokens) {
        auto it = idf.find(q);
        if (it == idf.end()) continue;
        score += it->second * BM25::term_frequency_score(item.bm25.tf_data, item.bm25.data_len, avg_len_, q);
    }
    return score;
}

std::vector<MemoryHit> MemoryManager::bm25_query(
    const std::vector<std::string> &tokens,
    const std::unordered_map<std::string, float> &idf,
    int top_k,
    float min_score) const {
    std::vector<MemoryHit> hits;
    hits.reserve(items_.size());
    for (const auto & [id, item] : items_) {
        float score = bm25_score(item, tokens, idf);
        if (score >= min_score)
            hits.push_back({id, item.data, score});
    }
    std::sort(hits.begin(), hits.end(), [](const MemoryHit & a, const MemoryHit & b) { return a.score > b.score; });
    if (static_cast<int>(hits.size()) > top_k) hits.resize(top_k);
    return hits;
}

std::vector<MemoryHit> MemoryManager::embed_query(const std::string &query, int top_k, float min_score) const {
    if (!embedder_) return {};
    auto qv = embedder_->embed(query);
    if (qv.empty()) return {};
    std::vector<MemoryHit> hits;
    for (const auto & [id, item] : items_) {
        if (item.embeddings.empty()) continue;
        float score = EmbeddingService::cosine_similarity(std::span<const float>(qv) , std::span<const float>(item.embeddings));
        if (score >= min_score)
            hits.push_back({id, item.data, score});
    }

    std::sort(hits.begin(), hits.end(), [](const MemoryHit & a, const MemoryHit & b) { return a.score > b.score; });
    if (static_cast<int>(hits.size()) > top_k) hits.resize(top_k);
    return hits;
}

std::vector<std::string> MemoryManager::deduplicated_tokens(const std::string &text) {
    auto tokens = BM25::tokenize(text);
    std::unordered_set<std::string> seen;
    tokens.erase(std::remove_if(tokens.begin(), tokens.end(), [&seen](const std::string & t) {
        return !seen.insert(t).second;
    }), tokens.end());
    return tokens;
}

std::unordered_map<std::string, float> MemoryManager::build_idf_cache(const std::vector<std::string> &tokens) const {
    std::unordered_map<std::string, float> cache;
    cache.reserve(tokens.size());
    const int N = static_cast<int>(items_.size());
    for (const auto & t : tokens) {
        int df = 0;
        if (auto it = doc_freq_.find(t); it != doc_freq_.end()) df = it->second;
        cache[t] = BM25::inverse_document_frequency(N, df);
    }
    return cache;
}

void MemoryManager::increase_avg_len(size_t new_len) {
    ++item_count_;
    avg_len_ += (static_cast<float>(new_len) - avg_len_) / static_cast<float>(item_count_);

}
