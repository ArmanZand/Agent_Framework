#include "PromptService.h"
#include "BackendService.h"
#include "../infrastructure/Stopwatch.h"
PromptService::PromptService(std::shared_ptr<ModelService> model_service, const PromptConfig &config) : model_(std::move(model_service)), config_(config) {
    init_template(model_->vocab());
}

std::vector<llama_token> PromptService::tokenize(const std::string &text, bool add_special, bool parse_special) const {
    auto & llama = BackendService::instance().llama();
    llama_vocab * vocab = model_->vocab();

    int n = llama.tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, add_special, parse_special);
    if (n == 0) return {};

    int capacity = (n < 0) ? -n : n;
    std::vector<llama_token> tokens(capacity);
    int actual = llama.tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), capacity, add_special, parse_special);

    if (actual < 0) {
        tokens.resize(-actual);
        actual = llama.tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), -actual, add_special, parse_special);
    }
    tokens.resize(static_cast<size_t>(actual > 0 ? actual : 0));
    return tokens;
}

std::string PromptService::detokenize(llama_token token, bool lstrip, bool special) const {
    auto & llama = BackendService::instance().llama();
    llama_vocab * vocab = model_->vocab();

    char buf[32];
    int n = llama.token_to_piece(vocab, token, buf, sizeof(buf), lstrip ? 1 : 0, special);
    if (n < 0) {
        std::string s(static_cast<size_t>(-n), '\0');
        llama.token_to_piece(vocab, token, s.data(), -n, lstrip ? 1 : 0, special);
        return s;
    }
    return std::string(buf, static_cast<size_t>(n));
}

std::string PromptService::detokenize(const std::vector<llama_token> &tokens) const {
    std::string result;
    result.reserve(tokens.size() * 4);
    for (llama_token t : tokens) {
        result += detokenize(t);
    }
    return result;
}

int PromptService::count_tokens(const std::string &text, bool add_special, bool parse_special) const {
    auto & llama = BackendService::instance().llama();
    int n = llama.tokenize(model_->vocab(), text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, add_special, parse_special);
    return (n < 0) ? -n : n;
}

int PromptService::count_chat_tokens(const std::vector<ChatMessage> & messages, bool add_assistant_turn) const {
    std::string rendered = render_chat(messages, add_assistant_turn);
    return count_tokens(rendered, false, true);
}


std::string PromptService::render_chat(const std::vector<ChatMessage> &messages, bool add_assistant_turn) const {
    auto& llama = BackendService::instance().llama();
    std::string out;
    for (const auto & m : messages) {
        out += render_turn(m.role, m.content, false);
    }
    if (add_assistant_turn) {
        out += "<start_of_turn>model\n";
    }
    return out;
}

// std::string PromptService::render_turn(const std::string &role, const std::string &content,
// bool add_assistant_turn) const {
//     std::string out = "<start_of_turn>" + role + "\n" + content + "<end_of_turn>\n";
//     if (add_assistant_turn)
//         out += "<start_of_turn>model\n";
//     return out;
// }

std::string PromptService::render_turn(const std::string &role, const std::string &content, bool add_assistant_turn) const {
    auto & llama = BackendService::instance().llama();
    const char * tmpl = chat_template_.empty() ? nullptr : chat_template_.c_str();
    std::vector<llama_chat_message> msg = {{role.c_str(), content.c_str()}};

    int len = llama.chat_apply_template(tmpl, msg.data(), msg.size(), add_assistant_turn, nullptr, 0);
    if (len < 0) {
        fprintf(stderr, "[PromptService] chat_apply_template size probe failed.\n");
        return {};
    }
    std::string buf(static_cast<size_t>(len + 1), '\0');
    llama.chat_apply_template(tmpl, msg.data(), msg.size(), add_assistant_turn, buf.data(), len + 1);
    buf.resize(static_cast<size_t>(len));
    return buf;
}

std::string PromptService::render_partial_assistant(const std::string &content) const {
    const std::string sentinel = "\x01\x02SENTINEL\x02\x01";
    std::string probed = render_turn(MessageRole::Assistant, sentinel, false);
    size_t pos = probed.find(sentinel);
    if (pos == std::string::npos)
        return render_turn(MessageRole::Assistant, content, false);
    return probed.substr(0, pos) + content;
}

// std::string PromptService::render_apply_template(const std::string &user_message, bool add_assistant_turn) const {
//     auto & llama = BackendService::instance().llama();
//     const char * tmpl = chat_template_.empty() ? nullptr : chat_template_.c_str();
//     std::vector<llama_chat_message> msg = {{RoleType::User, user_message.c_str()}};
//
//     int len = llama.chat_apply_template(tmpl, msg.data(), 1, add_assistant_turn, nullptr, 0);
//     if (len < 0) {
//         fprintf(stderr, "[PromptService] chat_apply_template size probe failed.\n");
//         return {};
//     }
//     std::string buf(static_cast<size_t>(len + 1), '\0');
//     llama.chat_apply_template(tmpl, msg.data(), 1, add_assistant_turn, buf.data(), len + 1);
//     buf.resize(static_cast<size_t>(len));
//     return buf;
// }


int PromptService::decode_batch(ContextService &ctx, const std::vector<llama_token> &tokens) const {
    if (tokens.empty()) return 0;
    auto & llama = BackendService::instance().llama();

    const int n_batch = ctx.n_batch();
    size_t offset = 0;
    while (offset < tokens.size()) {
        int32_t chunk = static_cast<int32_t>((std::min)(tokens.size() - offset, size_t(n_batch)));
        llama_batch batch = llama.batch_get_one(const_cast<llama_token*>(tokens.data() + offset), chunk);
        if (llama.decode(ctx.context(), batch) != 0)
            return -1;
        offset += chunk;
    }
    return 0;
}

std::string PromptService::generate(ContextService &ctx, SamplerService &sampler, const std::string &prompt,
                                    TokenCallback on_token, CompletionCallback on_done) const {
    auto & llama = BackendService::instance().llama();
    llama_vocab * vocab = model_->vocab();

    auto prompt_tokens = tokenize(prompt, config_.add_bos, config_.parse_special);
    if (prompt_tokens.empty()) return {};

    int ret = decode_batch(ctx, prompt_tokens);
    if (ret != 0) {
        fprintf(stderr, "[PromptService] decode_batch failed (ret=%d) on prompt.\n", ret);
        return {};
    }

    std::string response;
    response.reserve(256);


    for (int i = 0; config_.max_new_tokens <= 0 || i < config_.max_new_tokens; ++i) {
        llama_token token = sampler.sample(ctx.context(), -1);
        if (llama.vocab_is_eog(vocab, token)) break;

        std::string piece = detokenize(token, false, config_.parse_special);
        response += piece;

        if (on_token && !on_token(piece)) break;

        std::vector<llama_token> single = { token };
        ret = decode_batch(ctx, single);
        if (ret != 0) {
            if (ret == 1) {
                fprintf(stderr, "[PromptService] KV-cache full - stopping generation.\n");
            } else {
                fprintf(stderr, "[PromptService] decode_batch error (ret=%d).\n", ret);
            }
            break;
        }
    }

    if (on_done) on_done(response);
    return response;
}

std::string PromptService::generate_chat(ContextService &ctx, SamplerService &sampler,
    const std::vector<ChatMessage> &messages, TokenCallback on_token, CompletionCallback on_done) const {
    std::string prompt = render_chat(messages, true);
    return generate(ctx, sampler, prompt, std::move(on_token), std::move(on_done));
}

GenerationResult PromptService::generate_continue(ContextService &ctx, SamplerService &sampler, TokenCallback on_token) {
    Stopwatch sw;
    int token_count = 0;
    auto & llama = BackendService::instance().llama();
    llama_vocab * vocab = model_->vocab();

    std::string result;
    for (int i = 0; config_.max_new_tokens <= 0 || i < config_.max_new_tokens; ++i) {
        llama_token token = sampler.sample(ctx.context());

        if (token == llama.vocab_eos(vocab) || llama.vocab_is_eog(vocab, token))
            return { result, GenerationStop::Eos, token_count, sw.ms() };

        std::vector<char> buffer(128);
        int len = llama.token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
        if (len < 0) {
            buffer.resize(-len);
            len = llama.token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
        }
        std::string piece(buffer.data(), static_cast<size_t>(len));
        result += piece;
        ++token_count;

        if (on_token && !on_token(piece))
            return { result, GenerationStop::Cancelled, token_count, sw.ms() };

        llama_batch next_batch = llama.batch_get_one(&token, 1);
        if (llama.decode(ctx.context(), next_batch) != 0) {
            fprintf(stderr, "[PromptService] Failed to decode generated token - stopping.\n");
            return { result, GenerationStop::ContextFull, token_count, sw.ms() };
        }
    }
    return {result, GenerationStop::MaxTokens, token_count, sw.ms() };
}

std::vector<float> PromptService::generate_embedding(ContextService &ctx, const std::string &text) const {
    ctx.clear_kv_cache();

    auto tokens = tokenize(text, true, true);
    if (tokens.empty()) return {};

    int32_t ret = decode_batch(ctx, tokens);
    if (ret != 0) {
        fprintf(stderr, "[PromptService] generate_embedding decode failed (ret=%d).\n", ret);
        ctx.clear_kv_cache();
        return {};
    }

    std::vector<float> embedding;
    if (ctx.pooling_type() != LLAMA_POOLING_TYPE_NONE) {
        embedding = ctx.get_sequence_embedding(0);
    } else {
        embedding = ctx.get_token_embedding(static_cast<int32_t>(tokens.size()) - 1);
    }

    ctx.clear_kv_cache();
    return embedding;
}

std::vector<std::vector<float>> PromptService::generate_embeddings(ContextService &ctx,
    const std::vector<std::string> &texts) const {
    std::vector<std::vector<float>> results;
    results.reserve(texts.size());
    for (const auto & t : texts) {
        results.push_back(generate_embedding(ctx, t));
    }
    return results;
}

std::string PromptService::token_to_string(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buf(128);
    auto & llama = BackendService::instance().llama();
    int len = llama.token_to_piece(vocab, token, buf.data(), buf.size(), 0, true);
    if (len < 0) {
        buf.resize(-len);
        len = llama.token_to_piece(vocab, token, buf.data(), buf.size(), 0, true);
    }
    return std::string(buf.data(), len);
}

void PromptService::init_template(const llama_vocab *vocab) {
    auto & llama = BackendService::instance().llama();
    bos_token_str_ = token_to_string(vocab, llama.vocab_bos(vocab));
    eos_token_str_ = token_to_string(vocab, llama.vocab_eos(vocab));
    if (config_.get_default_template) {
        chat_template_ = model_->get_model_chat_template();
        if (chat_template_.empty()) {
            fprintf(stderr, "[PromptService] No chat template available.\n");
        }
    }
}
