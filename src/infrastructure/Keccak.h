#ifndef AGENT_FRAMEWORK_KECCAK_H
#define AGENT_FRAMEWORK_KECCAK_H
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>

enum KeccakParam {
    SHA3_224,
    SHA3_256,
    SHA3_384,
    SHA3_512,
    SHAKE128,
    SHAKE256,
};
class Keccak {
public:
    Keccak(KeccakParam param);
    void Absorb(std::span<const std::byte> data);
    bool Squeeze(std::span<uint8_t> output);
private:
    bool Init(const size_t capacity, const uint8_t delimited_suffix);
    static void XOF_F1600_ON_LANES(std::array<uint64_t, 25> & lanes);
    static void XOF_F1600(std::array<uint8_t, 200> & state);
    void XOF_FINALIZE();

    size_t rate_ = 0;
    size_t capacity_ = 0;
    size_t rate_in_bytes_ = 0;
    std::array<uint8_t,200> state_ = {};
    uint8_t delimited_suffix_ = 0;
    size_t pos_ = 0;
    bool finalized_ = false;
};


#endif //AGENT_FRAMEWORK_KECCAK_H