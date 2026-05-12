#ifndef AGENT_FRAMEWORK_STOPWATCH_H
#define AGENT_FRAMEWORK_STOPWATCH_H
#include <chrono>

struct Stopwatch {
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_ = Clock::now();
    void reset() { start_ = Clock::now() ;}
    double ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    }
};

#endif //AGENT_FRAMEWORK_STOPWATCH_H