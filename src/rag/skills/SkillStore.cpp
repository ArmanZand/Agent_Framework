
#include "SkillStore.h"

#include <fstream>

namespace fs = std::filesystem;


json SkillRecord::to_json() const {
    return  {
        {"id", id},
        {"name", name},
        {"description", description},
        {"embeddings", embeddings},
        {"trigger", trigger},
        {"source_file", source_file},
        {"content_hash", content_hash}
    };
}

SkillRecord SkillRecord::from_json(const json &j) {
    SkillRecord sr;
    if (j.contains("id")) sr.id = j["id"].get<std::string>();
    if (j.contains("name")) sr.name = j["name"].get<std::string>();
    if (j.contains("description")) sr.description = j["description"].get<std::string>();
    if (j.contains("embeddings")) sr.embeddings = j["embeddings"].get<std::vector<float>>();
    if (j.contains("trigger")) sr.trigger = j["trigger"].get<std::string>();
    if (j.contains("source_file")) sr.source_file = j["source_file"].get<std::string>();
    if (j.contains("content_hash")) sr.content_hash = j["content_hash"].get<std::string>();
    return sr;
}

SkillStore::SkillStore(std::shared_ptr<EmbeddingService> embedder)
    : embedder_(std::move(embedder)) {}

bool SkillStore::load_directory(const std::string &path) {
    skills_.clear();

    std::error_code ec;
    if (!fs::exists(path, ec) || !fs::is_directory(path, ec)) {
        fprintf(stderr, "[SkillStore] Not a directory: %s\n", path.c_str());
        return false;
    }

    int loaded = 0;
    for (auto & entry : fs::recursive_directory_iterator(path, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".md") continue;

        SkillRecord record;
        if (!parse_skill(entry.path().string(), record)) continue;

        if (skills_.count(record.id)) {
            fprintf(stderr, "[SkillStore] Duplicate id '%s' in %s, skipping.\n", record.id.c_str(), entry.path().string().c_str());
            continue;
        }
        record.content_hash = compute_hash(record);
        skills_.emplace(record.id, std::move(record));
        ++loaded;
    }
    fprintf(stdout, "[SkillStore] Loaded %d skill files from %s\n", loaded, path.c_str());

    if (embedder_ && loaded > 0) {
        std::string cache_path = (fs::path(path) / ".skill_embeddings.json").string();
        load_embedding_cache(cache_path);
        embed_missing();
        save_embedding_cache(cache_path);
    }
    return loaded > 0;
}

std::vector<SkillRecord> SkillStore::search_skills(const std::string &query, int top_k) const {
    if (skills_.empty() || !embedder_) return {};
    auto qv = embedder_->embed(query);
    std::span<const float> qspan = qv;

    std::vector<std::pair<float, std::string>> scores;
    for (auto & [id, record] : skills_) {
        if (record.embeddings.empty()) continue;
        float s = EmbeddingService::cosine_similarity(qspan, std::span<const float>(record.embeddings));
        scores.push_back({s, id});
    }
    std::sort(scores.begin(), scores.end(), [](const auto & a, const auto & b) { return a.first > b.first;});
    std::vector<SkillRecord> results;
    for (int i =0; i < (int)scores.size() && i < top_k; ++i)
        results.push_back(skills_.at(scores[i].second));
    return results;
}

std::string SkillStore::get_skill_content(const std::string &skill_id) const {
    auto it = skills_.find(skill_id);
    if (it == skills_.end()) return {};

    std::ifstream f(it->second.source_file);
    if (!f) {
        fprintf(stderr, "[SkillStore] Cannot open skill file: %s\n", it->second.source_file.c_str());
        return {};
    }

    std::string content((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());

    if (content.rfind("---", 0) == 0) {
        size_t end = content.find("\n---", 3);
        if (end != std::string::npos) content = content.substr(end + 4);
    }
    size_t first = content.find_first_not_of("\n\r");
    if (first != std::string::npos) content = content.substr(first);
    return content;
}

std::optional<SkillRecord> SkillStore::get_skill_record(const std::string &skill_id) const {
    auto it = skills_.find(skill_id);
    return (it == skills_.end()) ? std::nullopt : std::optional<SkillRecord>(it->second);
}

bool SkillStore::parse_skill(const std::string &file_path, SkillRecord &out) const {
    std::ifstream f(file_path);
    if (!f) {
        fprintf(stderr, "[SkillStore] Cannot open: %s\n", file_path.c_str());
        return false;
    }

    auto trim = [](const std::string & s) {
        size_t l = s.find_first_not_of(" \t\r\n");
        if (l == std::string::npos) return std::string {};
        size_t r = s.find_last_not_of(" \t\r\n");
        return s.substr(l, r - l + 1);
    };
    std::string line;
    if (!std::getline(f, line) || trim(line) != "---") {
        fprintf(stderr, "[SkillStore] %s: missing opening '---'\n", file_path.c_str());
        return false;
    }
    while (std::getline(f, line)) {
        if (trim(line) == "---") {
            out.source_file = file_path;
            if (out.id.empty() || out.trigger.empty()) {
                fprintf(stderr, "[SkillStore] %s: missing required field (skill or trigger)\n",
                    file_path.c_str());
                return false;
            }
            return true;
        }
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = trim(line.substr(0, colon));
        std::string value = trim(line.substr(colon + 1));
        if (key == "skill") out.id = value;
        else if (key == "name") out.name = value;
        else if (key == "description") out.description = value;
        else if (key == "trigger") out.trigger = value;
    }
    fprintf(stderr, "[SkillStore] %s: missing closing '---'\n", file_path.c_str());
    return false;
}

void SkillStore::embed_missing() {
    if (!embedder_) return;
    int embedded = 0;
    for (auto & [id, record] : skills_) {
        if (!record.embeddings.empty()) continue;
        std::string text = record.name +". " + record.description + ". " + record.trigger;
        auto vec = embedder_->embed(text);
        if (vec.empty()) {
            fprintf(stderr, "[SkillStore] Embeedding failed for '%s'\n", id.c_str());
            continue;
        }
        record.embeddings = std::move(vec);
        ++embedded;
    }
    if (embedded > 0)
        fprintf(stdout, "[SkillStore] Embedded %d new skills.\n", embedded);
}

bool SkillStore::load_embedding_cache(const std::string &cache_path) {
    std::ifstream f(cache_path);
    if (!f) {
        fprintf(stdout, "[SkillStore] No cache file found at %s\n", cache_path.c_str());
        return false;
    }

    json j;
    try { f >> j; } catch (...) {
        fprintf(stderr, "[SkillStore] Cache unreadable, ignoring.\n");
        return false;
    }

    if (j.value("embedder_model", "") != embedder_->id()) {
        fprintf(stderr, "[SkillStore] Cache model mismatch, ignoring cache.\n");
        return false;
    }

    int hits = 0;
    for (auto & e : j.value("entries", json::array())) {
        std::string id = e.value("id", "");
        std::string hash = e.value("hash", "");
        auto it = skills_.find(id);
        if (it == skills_.end()) continue;
        if (it->second.content_hash != hash) continue;
        if (!e.contains("vector") || !e["vector"].is_array()) continue;
        it->second.embeddings = e["vector"].get<std::vector<float>>();
        ++hits;
    }
    fprintf(stdout, "[SkillStore] %d cache embeddings reused.\n", hits);
    return true;
}

bool SkillStore::save_embedding_cache(const std::string &cache_path) const {
    json j;
    j["embedder_model"] = embedder_->id();
    j["entries"] = json::array();
    for (auto & [id, record] : skills_) {
        if (record.embeddings.empty()) continue;
        json e;
        e["id"] = id;
        e["hash"] = record.content_hash;
        e["vector"] = record.embeddings;
        j["entries"].push_back(std::move(e));
    }
    std::ofstream f(cache_path);
    if (!f) {
        fprintf(stderr, "[SkillStore] Cannot write cache to %s\n", cache_path.c_str());
        return false;
    }
    f << j.dump(2);
    fprintf(stdout, "[SkillStore] Saved embedding cache to %s\n", cache_path.c_str());
    return true;
}

std::string SkillStore::compute_hash(const SkillRecord &record) {
    std::string data = record.id + "\x1f" + record.name + "\x1f" + record.description + "\x1f" + record.trigger;
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : data) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return std::string(buf);
}


