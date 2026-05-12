#include "BM25.h"

#include <cmath>

float BM25::term_frequency_score(const std::unordered_map<std::string, int> &term_frequency_map, int term_len, float avg_len,
                                 const std::string &term, float k1, float b) {
    if (avg_len == 0.f) return 0.f;
    auto it = term_frequency_map.find(term);
    if (it == term_frequency_map.end()) return 0.f;
    const float norm = 1.0f - b + b * (static_cast<float>(term_len) / avg_len);
    return static_cast<float>(it->second) / (static_cast<float>(it->second) + k1 * norm);
}

float BM25::inverse_document_frequency(int total_documents, int local_frequency) {
    return std::log((static_cast<float>(total_documents) + 1.0f) / (static_cast<float>(local_frequency) + 1.0f)) + 1.0f;
    // const float idf = std::log((static_cast<float>(total_documents) - static_cast<float>(local_frequency) + 0.5f) /
    //     (static_cast<float>(local_frequency) + 0.5f));
    // return std::max(0.f, idf);
}

void BM25::extract_BM25_stats(const std::string &text, int &term_count,
    std::unordered_map<std::string, int> &term_frequency_map, std::unordered_set<std::string> &unique_terms) {
    auto tokens = tokenize(text);
    term_count = static_cast<int>(tokens.size());
    for (const auto & t : tokens) {
        term_frequency_map[t]++;
        unique_terms.insert(t);
    }
}

void BM25::extract_BM25_stats(const std::vector<std::string> &text_array, int &term_count,
    std::unordered_map<std::string, int> &term_frequency_map, std::unordered_set<std::string> &unique_terms) {
    for (const auto & text : text_array) {
        auto tokens = tokenize(text);
        term_count += static_cast<int>(tokens.size());
        for (const auto & t : tokens) {
            term_frequency_map[t]++;
            unique_terms.insert(t);
        }
    }
}

std::vector<std::string> BM25::tokenize(const std::string &text) {
    std::vector<std::string> out;
    std::string cur;
    for (const char ch : text) {
        auto uc = static_cast<unsigned char>(ch);
        if (std::isalnum(uc)) {
            cur += static_cast<char>(std::tolower(uc));
        } else {
            if (cur.size() >= 2) out.push_back(cur);
            cur.clear();
        }
    }
    if (cur.size() >= 2) out.push_back(cur);
    return out;
}
