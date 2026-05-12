#ifndef AGENT_FRAMEWORK_STREAMPROCESSOR_H
#define AGENT_FRAMEWORK_STREAMPROCESSOR_H

#include <string>
#include <vector>
#include <functional>
#include "../interfaces/ITool.h"
#include "../model_services/PromptService.h"

struct VocabularyConfig {
    std::string grammar = R"(
root    ::= (thought tool-call+)* final-response
final-response ::= [ \t\n]* visible-char [^\x00<]*
visible-char ::= [^ \t\n\r\x00<]
thought ::= [^\x00<]*
tool-call ::= "<tool_call>" space json space "</tool_call>" space
json   ::= "{" space "\"name\"" space ":" space string space "," space "\"arguments\"" space ":" space object space "}"
object ::= "{" space (string space ":" space value (space "," space string space ":" space value)*)? space "}"
array  ::= "[" space (value (space "," space value)*)? space "]"
value  ::= object | array | string | number | "true" | "false" | "null"
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\""
number ::= "-"? ("0" | [1-9][0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
space  ::= [ \t\n]*
)";
    std::string TOOL_OPEN   = "<tool_call>";
    std::string TOOL_CLOSE  = "</tool_call>";
    std::string THINK_OPEN  = "<think>";
    std::string THINK_CLOSE = "</think>";
};

using ToolStartCallback     = std::function<void(const ToolCall & call)>;
using ToolResultCallback    = std::function<void(const ToolCall & call, const ToolResult & result)>;
using ThinkingStartCallback = std::function<void()>;
using ThinkingEndCallback   = std::function<void()>;


struct AgentCallbacks {
    TokenCallback         on_token           = nullptr;
    TokenCallback         on_thinking_token  = nullptr;
    ThinkingStartCallback on_thinking_start  = nullptr;
    ThinkingEndCallback   on_thinking_end    = nullptr;
    ToolStartCallback     on_tool_start      = nullptr;
    ToolResultCallback    on_tool_result     = nullptr;
};

struct StreamState {
    enum class Phase { Streaming, BufferingToolCall, BufferingThink };
    Phase phase = Phase::Streaming;
    std::string pending_tag;
    std::string tool_call_buffer;
    std::string think_buffer;
    std::vector<ToolCall> completed_tool_calls;
    std::string visible_response;
    bool think_started = false;

    bool has_complete_call() const { return !completed_tool_calls.empty(); }
    void reset_for_next_pass() {
        visible_response.clear();
        completed_tool_calls.clear();
        tool_call_buffer.clear();
        think_buffer.clear();
        phase = Phase::Streaming;
        pending_tag.clear();
        think_started = false;
    }
};

class StreamProcessor {
public:
    StreamProcessor(const VocabularyConfig & vocab, const AgentCallbacks & callbacks);
    bool process(const std::string & token, StreamState & state);

private:
    struct OpenTagMatch {
        size_t position = std::string::npos;
        size_t open_len = 0;
        bool is_tool = false;
        bool found() const { return position != std::string::npos; }
    };

    bool process_streaming_phase(const std::string & token, StreamState & state);
    bool process_thinking_phase(const std::string & token, StreamState & state);
    bool process_tool_call_phase(const std::string & token, StreamState & state);

    void emit_visible(StreamState & state, const std::string & text);
    OpenTagMatch find_earliest_open_tag(const std::string & text) const;
    bool enter_tag_phase(StreamState & state, const OpenTagMatch & match);
    bool flush_or_retain_partial_tag(StreamState & state);


    const VocabularyConfig & vocab_;
    const AgentCallbacks & callbacks_;
};




#endif //AGENT_FRAMEWORK_STREAMPROCESSOR_H