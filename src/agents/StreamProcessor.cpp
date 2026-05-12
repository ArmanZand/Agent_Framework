#include "StreamProcessor.h"

StreamProcessor::StreamProcessor(const VocabularyConfig &vocab, const AgentCallbacks &callbacks) : vocab_(vocab), callbacks_(callbacks) {
}

bool StreamProcessor::process(const std::string &token, StreamState &state) {
    switch (state.phase) {
        case StreamState::Phase::Streaming: return process_streaming_phase(token, state);
        case StreamState::Phase::BufferingThink: return process_thinking_phase(token, state);
        case StreamState::Phase::BufferingToolCall: return process_tool_call_phase(token, state);
    }
    return true;
}

StreamProcessor::OpenTagMatch StreamProcessor::find_earliest_open_tag(const std::string &text) const {
    size_t tool_pos = text.find(vocab_.TOOL_OPEN);
    size_t think_pos = text.find(vocab_.THINK_OPEN);

    OpenTagMatch m;
    const bool has_tool = tool_pos != std::string::npos;
    const bool has_think = think_pos != std::string::npos;
    if (!has_tool && !has_think) return m;

    const bool tool_wins = has_tool && (!has_think || tool_pos < think_pos);
    m.position = tool_wins ? tool_pos : think_pos;
    m.open_len = tool_wins ? vocab_.TOOL_OPEN.size() : vocab_.THINK_OPEN.size();
    m.is_tool = tool_wins;
    return m;
}

bool StreamProcessor::enter_tag_phase(StreamState &state, const OpenTagMatch &match) {
    if (match.position > 0) {
        emit_visible(state, state.pending_tag.substr(0, match.position));
    }

    std::string remainder = state.pending_tag.substr(match.position + match.open_len);
    state.pending_tag.clear();
    state.phase = match.is_tool ? StreamState::Phase::BufferingToolCall : StreamState::Phase::BufferingThink;

    if (!remainder.empty()) return process(remainder, state);
    return true;
}

bool StreamProcessor::flush_or_retain_partial_tag(StreamState &state) {
    size_t last_lt = state.pending_tag.rfind('<');

    if (last_lt == std::string::npos) {
        emit_visible(state, state.pending_tag);
        state.pending_tag.clear();
        return true;
    }

    if (last_lt > 0) {
        emit_visible(state, state.pending_tag.substr(0, last_lt));
        state.pending_tag = state.pending_tag.substr(last_lt);
    }

    bool could_be_tag = vocab_.TOOL_OPEN.starts_with(state.pending_tag)
            || vocab_.THINK_OPEN.starts_with(state.pending_tag);

    if (!could_be_tag) {
        emit_visible(state, state.pending_tag);
        state.pending_tag.clear();
    }
    return true;
}

bool StreamProcessor::process_streaming_phase(const std::string &token, StreamState &state) {
    state.pending_tag += token;

    if (state.visible_response.empty() && state.completed_tool_calls.empty()) {
        size_t first = state.pending_tag.find_first_not_of(" \n\r\t");
        if (first == std::string::npos) return true;
        if (first > 0) state.pending_tag.erase(0, first);
    }

    auto match = find_earliest_open_tag(state.pending_tag);
    if (match.found()) return enter_tag_phase(state, match);

    return flush_or_retain_partial_tag(state);
}

bool StreamProcessor::process_thinking_phase(const std::string &token, StreamState &state) {
    if (!state.think_started) {
        if (callbacks_.on_thinking_start) callbacks_.on_thinking_start();
        state.think_started = true;
    }
    state.think_buffer += token;

    size_t tool_pos = state.think_buffer.find(vocab_.TOOL_OPEN);
    size_t close_pos = state.think_buffer.find(vocab_.THINK_CLOSE);

    if (tool_pos != std::string::npos &&
        (close_pos == std::string::npos || tool_pos < close_pos)) {
        std::string remainder = state.think_buffer.substr(tool_pos);
        state.think_buffer.clear();
        state.phase = StreamState::Phase::Streaming;
        if (callbacks_.on_thinking_end) callbacks_.on_thinking_end();
        state.think_started = false;
        return process(remainder, state);
        }

    if (close_pos != std::string::npos) {
        if (callbacks_.on_thinking_token && close_pos > 0)
            callbacks_.on_thinking_token(state.think_buffer.substr(0, close_pos));
        std::string remainder = state.think_buffer.substr(close_pos + vocab_.THINK_CLOSE.size());
        state.think_buffer.clear();
        state.phase = StreamState::Phase::Streaming;
        if (callbacks_.on_thinking_end) callbacks_.on_thinking_end();
        state.think_started = false;
        if (!remainder.empty()) return process(remainder, state);
        return true;
    }

    if (state.think_buffer.size() > vocab_.THINK_CLOSE.size()) {
        size_t safe_len = state.think_buffer.size() - vocab_.THINK_CLOSE.size();
        if (callbacks_.on_thinking_token)
            callbacks_.on_thinking_token(state.think_buffer.substr(0, safe_len));
        state.think_buffer = state.think_buffer.substr(safe_len);
    }

    return true;
}

bool StreamProcessor::process_tool_call_phase(const std::string &token, StreamState &state) {
    state.tool_call_buffer += token;
    size_t close_pos = state.tool_call_buffer.find(vocab_.TOOL_CLOSE);
    if (close_pos == std::string::npos) return true;

    std::string json_str  = state.tool_call_buffer.substr(0, close_pos);
    std::string remainder = state.tool_call_buffer.substr(close_pos + vocab_.TOOL_CLOSE.size());
    state.tool_call_buffer.clear();
    size_t l = json_str.find_first_not_of(" \n\r\t");
    size_t r = json_str.find_last_not_of(" \n\r\t");
    json_str = (l == std::string::npos) ? "" : json_str.substr(l, r - l + 1);

    if (!json_str.empty()) {
        try {
            json j = json::parse(json_str);
            ToolCall call;
            call.id   = "call_" + std::to_string(state.completed_tool_calls.size());
            call.name = j.value("name", "");

            if (j.contains("arguments")) {
                if (j["arguments"].is_string()) {
                    try { call.arguments = json::parse(j["arguments"].get<std::string>()); }
                    catch (...) {
                        call.arguments = json{
                                { "raw",   j["arguments"] },
                                { "error", "failed to parse string arguments" }
                        };
                    }
                } else {
                    call.arguments = j["arguments"];
                }
            }

            if (!call.name.empty()) {
                state.completed_tool_calls.push_back(call);
                if (callbacks_.on_tool_start) callbacks_.on_tool_start(call);
            }
        }
        catch (const std::exception & e) {
            ToolCall bad;
            bad.id        = "call_" + std::to_string(state.completed_tool_calls.size());
            bad.name      = "__malformed__";
            bad.arguments = { { "raw", json_str }, { "error", e.what() } };
            state.completed_tool_calls.push_back(bad);
            fprintf(stderr, "\n[StreamProcessor] Failed to parse tool call: %s\n", e.what());
        }
    }

    state.phase = StreamState::Phase::Streaming;
    if (!remainder.empty()) return process(remainder, state);
    return true;
}

void StreamProcessor::emit_visible(StreamState &state, const std::string &text) {
    if (text.empty()) return;
    if (callbacks_.on_token) callbacks_.on_token(text);
    state.visible_response += text;
}

