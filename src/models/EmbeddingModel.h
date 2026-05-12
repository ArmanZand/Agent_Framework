#ifndef AGENT_FRAMEWORK_EMBEDDINGMODEL_H
#define AGENT_FRAMEWORK_EMBEDDINGMODEL_H
#include <string>
#include <vector>

class EmbeddingModel {
public:
    enum class PoolingType {
        NONE,
        MEAN,
        CLS,
        LAST
    };
    enum class NormalizationType {
        NONE = -1,
        MAX_ABS = 0,
        EUCLIDEAN = 2,
        P_NORM = 3
    };
private:
    std::string model_path;
    int dimension;
    bool is_loaded;

    struct ModelState {
        void* llama_wrapper;
        void* model;
        void* vocab;
    } state;

    int n_ctx;
    int n_gpu_layers;
    bool verbose;
    NormalizationType norm_type;
    PoolingType pooling_type;
    int p_norm_value;

    void normalize_embedding(std::vector<float>& embedding);
public:
    struct Config {
        std::string model_path;
        int n_ctx = 512;
        int n_gpu_layers = 99;
        bool verbose = true;
        PoolingType pooling = PoolingType::NONE;
        NormalizationType normalization = NormalizationType::EUCLIDEAN;
        int p_norm_value = 2;
    };

    EmbeddingModel(const Config& cfg);
    ~EmbeddingModel();

    bool load();
    void unload();
    bool is_model_Loaded() const { return is_loaded; };

    std::vector<float> embed(const std::string & text);
    int get_dimension() const { return dimension; };
};


#endif //AGENT_FRAMEWORK_EMBEDDINGMODEL_H