#ifndef AGENT_FRAMEWORK_CARD_H
#define AGENT_FRAMEWORK_CARD_H

#include <string>
#include <unordered_set>
#include <vector>
#include <unordered_map>
struct Card {
    std::string id;
    std::string title;
    std::string topic;
    std::vector<std::string> triggers;
    std::string summary;
    std::string body_path;
    std::vector<float> embedding;
    std::string content_hash;
    struct {
        int title_len = 0;
        int summary_len = 0;
        int trigger_len = 0;
        std::unordered_map<std::string, int> tf_title;
        std::unordered_map<std::string, int> tf_summary;
        std::unordered_map<std::string, int> tf_trigger;
        std::unordered_set<std::string> unique_terms;
    } bm25;

};

struct CardHit {
    std::string id;
    std::string title;
    std::string summary;
    float score = 0.0f;
};

struct CardSummary {
    std::string id;
    std::string title;
    std::string summary;
};

#endif //AGENT_FRAMEWORK_CARD_H