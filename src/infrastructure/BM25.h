#ifndef AGENT_FRAMEWORK_BM25F_H
#define AGENT_FRAMEWORK_BM25F_H

#include <unordered_map>
#include <vector>
#include <string>
#include <unordered_set>

class BM25 {
public:
    static float term_frequency_score(const std::unordered_map<std::string, int> &term_frequency_map, int term_len, float avg_len, const std::string & term, float k1 = 1.25f, float b = 0.75f);

    static float inverse_document_frequency(int total_documents, int local_frequency);

    static void extract_BM25_stats(const std::string & text, int & term_count, std::unordered_map<std::string, int> & term_frequency_map, std::unordered_set<std::string>  &unique_terms);

    static void extract_BM25_stats(const std::vector<std::string> & text_array, int & term_count, std::unordered_map<std::string, int> & term_frequency_map, std::unordered_set<std::string>  &unique_terms);

    static std::vector<std::string> tokenize(const std::string &text);
private:

};


#endif //AGENT_FRAMEWORK_BM25F_H