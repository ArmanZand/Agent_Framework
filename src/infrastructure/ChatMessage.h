#ifndef AGENT_FRAMEWORK_CHATMESSAGE_H
#define AGENT_FRAMEWORK_CHATMESSAGE_H

#include <string>

namespace MessageRole {
    inline constexpr const char * System    = "system";
    inline constexpr const char * User      = "user";
    inline constexpr const char * Assistant = "assistant";
    inline constexpr const char * Tool      = "tool";
    inline constexpr const char * Skill     = "skill";
    inline constexpr const char * Knowledge = "knowledge";
    inline constexpr const char * Summary   = "summary";
}


struct ChatMessage {
    std::string role;
    std::string content;
};


#endif //AGENT_FRAMEWORK_CHATMESSAGE_H