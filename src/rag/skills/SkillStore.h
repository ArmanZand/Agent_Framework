#ifndef AGENT_FRAMEWORK_SKILLSTORE_H
#define AGENT_FRAMEWORK_SKILLSTORE_H
#include <memory>
#include <string>

#include "../../model_services/EmbeddingService.h"
#include "nlohmann/json.hpp"
using json = nlohmann::ordered_json;

struct SkillRecord {
    std::string id;
    std::string name;
    std::string description;
    std::string trigger;
    std::string source_file;
    std::string content_hash;
    std::vector<float> embeddings;
    json to_json() const;
    static SkillRecord from_json(const json & j);
};
class SkillStore {
public:
    SkillStore(std::shared_ptr<EmbeddingService> embedder);

    bool load_directory(const std::string & path);

    std::vector<SkillRecord> search_skills(const std::string & query, int top_k = 10) const;
    std::string get_skill_content(const std::string & skill_id) const;
    std::optional<SkillRecord> get_skill_record(const std::string & skill_id) const;
    const std::map<std::string, SkillRecord> & all_skills() const { return skills_; }
    size_t size() const { return skills_.size(); }
private:
    bool parse_skill(const std::string & file_path, SkillRecord & out) const;
    void embed_missing();
    bool load_embedding_cache(const std::string & cache_path);
    bool save_embedding_cache(const std::string & cache_path) const;
    static std::string compute_hash(const SkillRecord & record);

    std::map<std::string, SkillRecord> skills_;
    std::shared_ptr<EmbeddingService> embedder_;
};


#endif //AGENT_FRAMEWORK_SKILLSTORE_H