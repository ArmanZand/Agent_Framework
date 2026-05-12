#ifndef AGENT_FRAMEWORK_UTILITY_H
#define AGENT_FRAMEWORK_UTILITY_H
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "Keccak.h"
#include <filesystem>
namespace fs = std::filesystem;
class Utility {
public:
    static std::string compute_hash(const std::string & data, int squeeze_byte_len = 12) {
        auto k = Keccak(KeccakParam::SHA3_224);
        k.Absorb(std::as_bytes(std::span(data)));
        std::vector<uint8_t> out(squeeze_byte_len);
        k.Squeeze(out);
        return to_hex(out);
    };

    static std::string to_hex(const std::vector<uint8_t>& data) {
        std::ostringstream oss;
        for (auto b : data) {
            oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
        }
        return oss.str();
    }
    static bool validate_dir(const std::string & dir_path) {
        std::error_code ec;
        if (fs::exists(dir_path, ec)) {
            return fs::is_directory(dir_path, ec);
        }
        fs::create_directories(dir_path, ec);
        if (ec) {
            fprintf(stderr, "[Utility] Cannot create directory %s: %s\n",
                    dir_path.c_str(), ec.message().c_str());
            return false;
        }
        return true;
    };
};

#endif //AGENT_FRAMEWORK_UTILITY_H