#include "KnowledgeBase.h"

#include <sstream>
#include <filesystem>
#include <fstream>
#include <set>
#include <unordered_set>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;
namespace {
    std::string trim(const std::string & s) {
        size_t l = s.find_first_not_of(" \t\r\n");
        if (l == std::string::npos) return {};
        size_t r = s.find_last_not_of(" \t\r\n");
        return s.substr(l, r- l + 1);
    }

    std::string to_lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }
    std::vector<std::string> split_csv(const std::string & s) {
        std::vector<std::string> out;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',')) {
            std::string t = trim(item);
            if (!t.empty()) out.push_back(t);
        }
        return out;
    }
}


KnowledgeBase::KnowledgeBase(std::shared_ptr<EmbeddingService> embedder): embedder_(std::move(embedder)) {}

bool KnowledgeBase::load_directory(const std::string &path) {
    cards_.clear();
    topic_to_ids_.clear();
    document_frequency.clear();
    total_cards_ = 0;

    std::error_code ec;
    if (!fs::exists(path, ec) || !fs::is_directory(path, ec)) {
        fprintf(stderr, "[KnowledgeBase] %s is not a directory\n", path.c_str());
        return false;
    }

    int loaded = 0;
    for (auto & e : fs::recursive_directory_iterator(path, ec)) {
        if (ec) break;
        if (!e.is_regular_file()) continue;
        if (e.path().extension() != ".md") continue;
        Card c;
        if (!parse_card(e.path().string(), c)) continue;
        if (cards_.count(c.id)) {
            fprintf(stderr, "[KnowledgeBase] duplicate id '%s' in %s, skipping\n",
                c.id.c_str(), e.path().string().c_str());
            continue;
        }
        c.content_hash = compute_hash(c);
        index_card(c);
        cards_.emplace(c.id, std::move(c));
        ++loaded;
    }
    total_cards_ = loaded;
    fprintf(stdout, "[KnowledgeBase] Loaded %d cards from %s\n", loaded, path.c_str());

    if (embedder_ && total_cards_ > 0) {
        std::string cache_path = (fs::path(path) / ".knowledge_embeddings.json").string();
        load_embedding_cache(cache_path);

        prepare_cards_bm25();
        rebuild_doc_frequency();
        compute_avg_field_lengths();

        embed_missing();
        save_embedding_cache(cache_path);
    }
    return loaded > 0;
}

std::string KnowledgeBase::render_topic_tree() const {
    std::map<std::string, int> counts;
    for (auto & [t, ids] : topic_to_ids_) {
        std::string acc;
        std::stringstream ss(t);
        std::string seg;
        while (std::getline(ss, seg, '/')) {
            if (!acc.empty()) acc += "/";
            acc += seg;
            counts[acc] += (int)ids.size();
        }
    }

    std::ostringstream os;
    os << "## Knowledge base\n";
    if (counts.empty()) {
        os << "(no knowledge cards loaded)\n";
        return os.str();
    }
    for (auto & [t, n] : counts) os << "- " << t << " (" << n << " cards)\n";
    os << "\nUse list_knowledge(topic) to browse, "
       << "search_knowledge(query) to find by keyword, "
       << "load_knowledge(id) to read a card's full content.\n";
    return os.str();
}


std::vector<CardHit> KnowledgeBase::search(const std::string &query, int max_results, float min_score) const {
    if (!embedder_ || query.empty()) return {};

    const int candidate_k = max_results * 3;
    auto bm25_hits = bm25_query(query, candidate_k, bm25_min_score_);
    auto embed_hits = embed_query(query, candidate_k, embed_min_score_);

    std::unordered_map<std::string, float> rrf;
    rrf.reserve(bm25_hits.size() + embed_hits.size());
    auto fuse = [&](const std::vector<CardHit> & hits, float weight) {
        for (size_t i = 0; i < hits.size();++i)
            rrf[hits[i].id] += weight / (60.0f + static_cast<float>(i));
    };
    fuse(bm25_hits, bm25_weight_);
    fuse(embed_hits, embed_weight_);

    std::vector<std::pair<std::string, float>> ranked(rrf.begin(), rrf.end());
    std::sort(ranked.begin(), ranked.end(),
        [](const auto & a, const auto & b) { return a.second > b.second; });

    if (!ranked.empty() && ranked.front().second > 0.f) {
        float inv = 1.0f / ranked.front().second;
        for (auto & [id, s] : ranked) s *= inv;
    }

    std::vector<CardHit> out;
    for (auto & [id, score] : ranked) {
        if (score < min_score) break;
        auto it = cards_.find(id);
        if (it == cards_.end()) continue;
        out.push_back({id, it->second.title, it->second.summary, score});
        if ((int)out.size() >= max_results) break;
    }
    return out;
}

std::vector<CardSummary> KnowledgeBase::list(const std::string &topic, const std::string &query) const {
    std::vector<CardSummary> out;
    auto add = [&](const Card & c) {
        out.push_back({ c.id, c.title, c.summary });
    };
    if (topic.empty()) {
        for (auto & [id, card] : cards_) add(card);
    } else {
        for (auto & [t, ids] : topic_to_ids_) {
            if (t == topic || t.rfind(topic + "/", 0) == 0) {
                for (auto & id : ids) {
                    auto it = cards_.find(id);
                    if (it != cards_.end()) add(it->second);
                }
            }
        }
    }

    if (!query.empty()) {
        auto terms = BM25::tokenize(query);
        std::vector<CardSummary> filtered;
        for (auto & cs : out) {
            auto it = cards_.find(cs.id);
            if (it == cards_.end()) continue;
            if (bm25f_score(it->second, terms) > 0.0f) filtered.push_back((cs));
        }
        out = std::move(filtered);
    }
    std::sort(out.begin(), out.end(),
        [](const CardSummary & a, const CardSummary & b) { return a.id < b.id; });
    return out;
}

std::optional<std::string> KnowledgeBase::load_body(const std::string &id) {
    auto it = cards_.find(id);
    if (it == cards_.end()) return std::nullopt;
    std::ifstream f(it->second.body_path);
    if (!f) return std::nullopt;

    std::string line;
    if (!std::getline(f, line) || trim(line) != "---") return std::nullopt;
    while (std::getline(f, line)) {
        if (trim(line) == "---") break;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool KnowledgeBase::has_card(const std::string &id) const {
    return cards_.find(id) != cards_.end();
}

bool KnowledgeBase::parse_card(const std::string & file_path, Card & out) const {
    std::ifstream f(file_path);
    if (!f) {
        fprintf(stderr, "[KnowledgeBase] cannot open %s\n", file_path.c_str());
        return false;
    }

    std::string line;
    if (!std::getline(f, line) || trim(line) != "---") {
        fprintf(stderr, "[KnowledgeBase] %s: missing opening '---'\n", file_path.c_str());
        return false;
    }

    while (std::getline(f, line)) {
        if (trim(line) == "---") {
            out.body_path = file_path;
            if (out.id.empty() || out.title.empty() || out.triggers.empty()) {
                fprintf(stderr, "[KnowledgeBase] %s: missing required field "
                    "(id, title, or triggers)\n", file_path.c_str());
                return false;
            }
            return true;
        }

        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = to_lower(line.substr(0, colon));
        std::string value = trim(line.substr(colon + 1));
        if (key == "id") out.id = value;
        else if (key == "title") out.title = value;
        else if (key == "topic") out.topic = value;
        else if (key == "triggers") out.triggers = split_csv(value);
        else if (key == "summary") out.summary = value;
    }
    fprintf(stderr, "[KnowledgeBase] %s: missing closing '---'\n", file_path.c_str());
    return false;
}

void KnowledgeBase::index_card(const Card &card) {
    topic_to_ids_[card.topic].push_back(card.id);
}


std::vector<CardHit> KnowledgeBase::bm25_query(const std::string &text, const int top_k, float min_score) const {
    const auto tokens = BM25::tokenize(text);
    std::vector<CardHit> hits;
    for (const auto & [id, card] : cards_) {
        float score = bm25f_score(card, tokens);
        hits.push_back({id, card.title, card.summary, score});
    }
    hits.erase(std::ranges::remove_if(hits, [&](const CardHit & c) {
        return c.score < min_score;
    }).begin(), hits.end());

    std::sort(hits.begin(), hits.end(), [](const CardHit & a, const CardHit & b) { return a.score > b.score; });
    if (static_cast<int>(hits.size()) > top_k) hits.resize(top_k);
    return hits;
}

std::vector<CardHit> KnowledgeBase::embed_query(const std::string &text, int top_k, float min_score) const {
    if (!embedder_) return {};
    auto qv = embedder_->embed(text);
    if (qv.empty()) return {};
    std::vector<CardHit> hits;
    for (const auto & [id, card] : cards_) {
        if (card.embedding.empty()) continue;
        float score = EmbeddingService::cosine_similarity(std::span<const float>(qv), std::span<const float>(card.embedding));
        hits.push_back({id, card.title, card.summary, score});
    }
    hits.erase(std::ranges::remove_if(hits, [&](const CardHit & c) {
        return c.score < min_score;
    }).begin(), hits.end());

    std::sort(hits.begin(), hits.end(), [](const CardHit & a, const CardHit & b) { return a.score > b.score; });
    if (static_cast<int>(hits.size()) > top_k) hits.resize(top_k);
    return hits;
}

std::string KnowledgeBase::compute_hash(const Card &card) {
    std::string data = card.title + "\x1f"+ card.summary + "\x1f";
    for (auto & t : card.triggers) { data += t; data += "\x1f"; }

    uint64_t h = 1469598103934665603ULL;
    for (unsigned char ch : data) {
        h ^= ch;
        h *= 1099511628211ULL;
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return std::string(buf);
}

void KnowledgeBase::embed_missing() {
    if (!embedder_) return;
    int embedded = 0;
    for (auto & [id, card] : cards_) {
        if (!card.embedding.empty()) continue;
        std::string text = card.title + ". " + card.summary + ". ";
        for (auto & t : card.triggers) { text += t; text += ", "; }

        auto vec = embedder_->embed(text);
        if (vec.empty()) {
            fprintf(stderr, "[KnowledgeBase] embed failed for '%s'\n", id.c_str());
            continue;
        }
        card.embedding = std::move(vec);
        ++embedded;
    }
    if (embedded > 0)
        fprintf(stdout, "[KnowledgeBase] Embedded %d new cards.\n", embedded);
}

void KnowledgeBase::prepare_cards_bm25() {
    for (auto & [id, card] : cards_) {
        if (!card.bm25.tf_title.empty() ||
            !card.bm25.tf_summary.empty() ||
            !card.bm25.tf_trigger.empty()) {
            continue;
        }

        update_card_bm25_stats(card);
    }
}

void KnowledgeBase::rebuild_doc_frequency() {
    document_frequency.clear();

    for (const auto & [id, card] : cards_) {
        for (const auto & term : card.bm25.unique_terms) {
            document_frequency[term]++;
        }
    }
}

void KnowledgeBase::compute_avg_field_lengths() {
    float total_title = 0.f;
    float total_summary = 0.f;
    float total_trigger = 0.f;

    for (const auto & [id, card] : cards_) {
        total_title += card.bm25.title_len;
        total_summary += card.bm25.summary_len;
        total_trigger += card.bm25.trigger_len;
    }

    int n = static_cast<int>(cards_.size());
    if (n > 0) {
        avg_title_len_ = total_title / n;
        avg_summary_len_ = total_summary / n;
        avg_trigger_len_ = total_trigger / n;
    }
    fprintf(stdout, "[KnowledgeBase] Computed BM25 average field lengths (ti: %.2f su: %.2f tr: %.2f).\n", avg_title_len_,avg_summary_len_, avg_trigger_len_);
}


bool KnowledgeBase::load_embedding_cache(const std::string &cache_path) {
    std::ifstream f(cache_path);;
    if (!f) {
        fprintf(stdout, "[KnowledgeBase] No cache file found at %s\n", cache_path.c_str());
        return false;
    }

    json j;
    try { f >> j; }
    catch (...) {
        fprintf(stderr, "[KnowledgeBase] cache file unreadable, ignoring\n");
        return false;
    }

    std::string cache_model = j.value("embedder_model", "");
    if (cache_model != embedder_->id()) {
        fprintf(stdout, "[KnowledgeBase] cache model '%s' != current '%s', ignoring cache\n",
            cache_model.c_str(), embedder_->id().c_str());
        return false;
    }

    int hits = 0;
    if (j.contains("entries") && j["entries"].is_array()) {
        for (auto & e : j["entries"]) {
            std::string id = e.value("id", "");
            std::string hash = e.value("hash", "");
            auto it = cards_.find(id);
            if (it == cards_.end()) continue;
            auto & c = it->second;
            if (c.content_hash != hash) continue;
            if (!e.contains("vector") || !e["vector"].is_array()) continue;
            std::vector<float> v;
            v.reserve(e["vector"].size());
            for (auto & x : e["vector"]) v.push_back(x.get<float>());
            c.embedding = std::move(v);

            if (e.contains("bm25") && e["bm25"].is_object()) {
                auto & bm25 = e["bm25"];

                c.bm25.title_len = bm25.value("title_len", 0);
                c.bm25.summary_len = bm25.value("summary_len", 0);
                c.bm25.trigger_len = bm25.value("trigger_len", 0);

                if (bm25.contains(("tf_title")))
                    c.bm25.tf_title = bm25["tf_title"].get<std::unordered_map<std::string, int>>();

                if (bm25.contains("tf_summary"))
                    c.bm25.tf_summary = bm25["tf_summary"].get<std::unordered_map<std::string, int>>();

                if (bm25.contains("tf_trigger"))
                    c.bm25.tf_trigger = bm25["tf_trigger"].get<std::unordered_map<std::string, int>>();

                if (bm25.contains("seen_terms")) {
                    for (auto & t : bm25["seen_terms"]) {
                        c.bm25.unique_terms.insert(t.get<std::string>());
                    }
                }

            }

            ++hits;

        }
    }

    fprintf(stdout, "[KnowledgeBase] %d cached embeddings reused.\n", hits);
    return true;

}

bool KnowledgeBase::save_embedding_cache(const std::string &cache_path) const {
    json j;
    j["embedder_model"] = embedder_->id();
    j["entries"] = json::array();
    for (auto & [id, card] : cards_) {
        if (card.embedding.empty()) continue;
        json e;
        e["id"] = id;
        e["hash"] = card.content_hash;
        e["vector"] = card.embedding;

        e["bm25"] = {
            {"title_len", card.bm25.title_len},
            {"summary_len", card.bm25.summary_len},
            {"trigger_len", card.bm25.trigger_len},
            {"tf_title", card.bm25.tf_title},
            {"tf_summary", card.bm25.tf_summary},
            {"tf_trigger", card.bm25.tf_trigger},
            {"seen_terms", std::vector<std::string>(card.bm25.unique_terms.begin(), card.bm25.unique_terms.end())},
        };

        j["entries"].push_back(std::move(e));


    }
    std::ofstream f(cache_path);
    if (!f) {
        fprintf(stderr, "[KnowledgeBase] cannot write cache to %s\n", cache_path.c_str());
        return false;
    }
    f << j.dump(2);
    fprintf(stdout, "[KnowledgeBase] Saved embedding cache to %s\n", cache_path.c_str());
    return true;
}

float KnowledgeBase::bm25f_score(const Card &card, const std::vector<std::string> &query_terms) const {
    if (query_terms.empty()) return 0.0f;

    constexpr float w_title = 2.0f;
    constexpr float w_summary = 1.0f;
    constexpr float w_trigger = 3.0f;

    float score = 0.0f;

    for (const auto & q : query_terms) {
        int df = 0;
        auto it = document_frequency.find(q);
        if (it != document_frequency.end()) df = it->second;

        float idf = BM25::inverse_document_frequency(total_cards_, df);

        float tf_title = BM25::term_frequency_score(card.bm25.tf_title, card.bm25.title_len, avg_title_len_, q);
        float tf_summary = BM25::term_frequency_score(card.bm25.tf_summary, card.bm25.summary_len, avg_summary_len_, q);
        float tf_trigger = BM25::term_frequency_score(card.bm25.tf_trigger, card.bm25.trigger_len, avg_trigger_len_, q);

        float tf_combined =
            w_title * tf_title +
            w_summary * tf_summary +
            w_trigger * tf_trigger;

        score += idf * tf_combined;
    }
    return score;
}

void KnowledgeBase::update_card_bm25_stats(Card &card) {
    //Card Title
    card.bm25 = {};
    BM25::extract_BM25_stats(card.title, card.bm25.title_len, card.bm25.tf_title, card.bm25.unique_terms);
    BM25::extract_BM25_stats(card.summary, card.bm25.summary_len, card.bm25.tf_summary, card.bm25.unique_terms);
    BM25::extract_BM25_stats(card.triggers, card.bm25.trigger_len, card.bm25.tf_trigger, card.bm25.unique_terms);
}

