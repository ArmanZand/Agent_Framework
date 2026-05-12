#ifndef AGENT_FRAMEWORK_AGENTSTATS_H
#define AGENT_FRAMEWORK_AGENTSTATS_H

#include <string>
#include <vector>
#include <sstream>
#include <ios>
#include <iomanip>
#include <iostream>

struct BoxPrinter {
    static void print(const std::vector<std::string> & lines, const std::string & color = "\033[1;36m") {
        if (lines.empty()) return;
        const char * reset = "\033[0m";

        size_t max_width = 0;
        for (const auto & line : lines)
            max_width = (std::max)(max_width, line.size());


        auto repeat = [](std::string s, size_t n) {
            std::string result;
            result.reserve(s.length() * n);
            for (size_t i =0 ; i < n;++i) {
                result += s;
            }
            return result;
        };

        //Top
        std::cout << "\n" << color << "+"
                   << repeat("-", max_width + 2)
                  << "+" << reset << "\n";

        //Line
        for (const auto & line : lines) {
            std::cout << color << "| "
                      << std::left << std::setw(max_width)
                      << line
                      << " |" << reset << "\n";
        }
        //Bottom
        std::cout << color << "+"
                  << repeat("-", max_width + 2)
                  << "+" << reset << "\n";
    };
};

struct ContextStats {
    int capacity = 0;
    int used = 0;
    int remaining = 0;
    int system_tokens = 0;
    int skills_tokens = 0;
    int summary_tokens = 0;

    int tail_messages = 0;
    int tail_tokens = 0;
    int tail_tool_tokens = 0;
    int tail_assistant_tokens = 0;
    int tail_user_tokens =0;

    int archive_messages = 0;
    int archive_tokens = 0;

    float usage_percent() const {
        return capacity > 0 ? 100.0f * used / capacity : 0.0f;
    }
    void print() const {
        std::vector<std::string> lines;
        lines.emplace_back("CONTEXT STATS");
        lines.emplace_back("");
        std::ostringstream oss;
        oss << "TOTAL:    " << used << "/" << capacity
            << " (" << std::fixed << std::setprecision(1)
            << usage_percent() << "%) remaining=" << remaining;
        lines.push_back(oss.str());
        oss.str(""); oss.clear();

        oss << "FIXED:    system=" << system_tokens
            << "  skills=" << skills_tokens
            << "  summary=" << summary_tokens;
        lines.push_back(oss.str());
        oss.str(""); oss.clear();

        oss << "TAIL:     " << tail_messages << " msgs / "
             << tail_tokens << " tokens";
        lines.push_back(oss.str());
        oss.str(""); oss.clear();

        oss << "          (tool=" << tail_tool_tokens
            << "  assistant=" << tail_assistant_tokens
            << "  user=" << tail_user_tokens << ")";
        lines.push_back(oss.str());
        oss.str(""); oss.clear();

        oss << "ARCHIVE:  " << archive_messages
            << " msgs / " << archive_tokens << " tokens";
        lines.push_back(oss.str());

        BoxPrinter::print(lines);
    }
};

struct ToolCallStats {
    std::string name;
    bool success = false;
    double exec_ms = 0;
    double encode_ms = 0;
};

struct GenerationPassStats {
    int tokens = 0;
    double ms = 0;
    double token_per_sec() const { return ms > 0 ? tokens * 1000.0 / ms : 0; }
};

struct TurnStats {
    std::vector<GenerationPassStats> generation_passes;
    std::vector<ToolCallStats> tool_calls;
    int context_full_events = 0;
    int results_stubbed = 0;
    int tick_events = 0;
    int compressions = 0;
    double flush_ms = 0;
    double total_ms = 0;
    ContextStats context_start, context_end;
    int total_tokens_generated() const {
        int n = 0;
        for (auto & p : generation_passes)
            n += p.tokens;
        return n;
    }
    void print() const {
        std::vector<std::string> lines;
        lines.emplace_back("TURN STATS");
        lines.emplace_back("");

        std::ostringstream oss;
        oss << "  total=" << std::fixed << std::setprecision(1) << total_ms << "ms"
            << "  tokens=" << total_tokens_generated()
            << "  tok/s avg=" << total_tokens_generated() * 1000 /total_ms << " tok/s";
        oss << "  flush=" << std::setprecision(1) << flush_ms << "ms";
        lines.push_back(oss.str());

        for (size_t i = 0; i < generation_passes.size(); ++i) {
            const auto & p = generation_passes[i];
            oss.str(""); oss.clear();
            oss << "  GEN[" << i << "] " << p.tokens << " tokens"
                << "  " << std::setprecision(1) << p.ms << "ms"
                << "  " << std::setprecision(1) << p.token_per_sec() << " tok/s";
            lines.push_back(oss.str());
        }

        if (!tool_calls.empty()) {
            lines.emplace_back("");
            for (const auto & tc : tool_calls) {
                oss.str(""); oss.clear();
                oss << "  TOOL " << std::left << std::setw(24) << tc.name
                    << (tc.success ? " OK  " : " FAIL")
                    << "  exec=" << std::fixed << std::setprecision(1) << tc.exec_ms << "ms"
                    << "  encode=" << std::setprecision(1) << tc.encode_ms << "ms";
                lines.push_back(oss.str());
            }
        }

        if (context_full_events > 0 || tick_events > 0 || compressions > 0 || results_stubbed > 0) {
            lines.emplace_back("");
            oss.str(""); oss.clear();
            oss << "  Events  ctx full=" << context_full_events
                << "  ticks=" << tick_events
                << "  compressions=" << compressions
                << "  stubbed=" << results_stubbed;
            lines.push_back(oss.str());
        }

        lines.emplace_back("");
        oss.str(""); oss.clear();
        int delta = context_end.used - context_start.used;
        double delta_percent = context_end.usage_percent() - context_start.usage_percent();
        oss << "  DELTA CTX  " << context_start.used << " -> " << context_end.used
            << " / " << context_end.capacity
            << "  (" << std::setprecision(1) << context_end.usage_percent() << "%)"
            << "   " << (delta >= 0 ? "+" : "") << delta << " (" << std::setprecision(1) << delta_percent << "%)";
        lines.push_back(oss.str());

        BoxPrinter::print(lines, "\033[1;33m");

    }
};


struct AgentStats {
    TurnStats last_turn;
    int total_turns = 0;
    int total_tokens = 0;
    int total_tool_calls = 0;
    int total_context_full = 0;
    int total_compressions = 0;

    void accumulate_turn_stats(const TurnStats & stats) {
        last_turn = stats;
        total_turns++;
        total_tokens += stats.total_tokens_generated();
        total_tool_calls += (int)stats.tool_calls.size();
        total_context_full += stats.context_full_events;
        total_compressions += stats.compressions;
    }

    void print_turn() const {
        last_turn.print();
    }

    void print_cumulative() const {
        std::vector<std::string> lines;
        std::ostringstream oss;

        lines.emplace_back("SESSION STAT TOTALS");
        lines.emplace_back("");

        oss << "  Turns:        " << total_turns;
        lines.push_back(oss.str()); oss.str(""); oss.clear();

        oss << "  Tokens:       " << total_tokens;
        lines.push_back(oss.str()); oss.str(""); oss.clear();

        oss << "  Tool calls:   " << total_tool_calls;
        lines.push_back(oss.str()); oss.str(""); oss.clear();

        oss << "  Ctx full:     " << total_context_full;
        lines.push_back(oss.str()); oss.str(""); oss.clear();

        oss << "  Compressions: " << total_compressions;
        lines.push_back(oss.str());

        BoxPrinter::print(lines, "\033[1;32m");
    }
};
#endif //AGENT_FRAMEWORK_AGENTSTATS_H