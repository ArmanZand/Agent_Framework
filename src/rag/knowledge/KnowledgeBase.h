#ifndef AGENT_FRAMEWORK_KNOWLEDGEBASE_H
#define AGENT_FRAMEWORK_KNOWLEDGEBASE_H
#include <map>
#include <memory>
#include <optional>
#include <string>
#include "Card.h"
#include "../../model_services/EmbeddingService.h"
#include "../../infrastructure/BM25.h"
class KnowledgeBase {
public:
    KnowledgeBase(std::shared_ptr<EmbeddingService> embedder = nullptr);

    bool load_directory(const std::string & path);

    std::string render_topic_tree() const;
    std::vector<CardHit> search(const std::string & query, int max_results = 10, float min_score = 0.7f) const;
    std::vector<CardSummary> list(const std::string & topic, const std::string & query = "") const;
    std::optional<std::string> load_body(const std::string & id);

    bool has_card(const std::string & id) const;
    size_t size() const { return cards_.size(); }

private:
    bool parse_card(const std::string & file_path, Card & out) const;
    void index_card(const Card & card);
    void embed_missing();
    void prepare_cards_bm25();
    void rebuild_doc_frequency();
    void compute_avg_field_lengths();
    void update_card_bm25_stats(Card & card);
    bool load_embedding_cache(const std::string & cache_path);
    bool save_embedding_cache(const std::string & cache_path) const;

    std::vector<CardHit> bm25_query(const std::string &text, int top_k, float min_score) const;
    std::vector<CardHit> embed_query(const std::string & text, int top_k, float min_score) const;

    static std::string compute_hash(const Card & card);


    std::map<std::string, Card> cards_;
    std::map<std::string, std::vector<std::string>> topic_to_ids_;
    int total_cards_ = 0;

    std::shared_ptr<EmbeddingService> embedder_;

    float bm25f_score(const Card & card, const std::vector<std::string> & query_terms) const;

    std::unordered_map<std::string, int> document_frequency;

    float embed_min_score_ = 0.7f;
    float bm25_min_score_ = 0.7f;
    float embed_weight_ = 0.7f;
    float bm25_weight_ = 0.3f;
    float avg_title_len_ = 0.f;
    float avg_summary_len_ = 0.f;
    float avg_trigger_len_ = 0.f;
};


#endif //AGENT_FRAMEWORK_KNOWLEDGEBASE_H