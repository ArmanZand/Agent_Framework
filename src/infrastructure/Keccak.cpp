#include "Keccak.h"

#include <stdexcept>

static constexpr uint8_t KECCAK_RHO_OFFSETS[24] = {
    1,  3,  6, 10, 15, 21,
    28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43,
    62, 18, 39, 61, 20, 44
};
static constexpr uint8_t KECCAK_PI_INDEX[24] = {
    2, 11,  7, 13, 18, 15,
    1,  8, 16,  9, 24, 20,
    3, 19, 23, 17, 12, 10,
    4, 22, 14, 21,  6,  5
};
static constexpr uint8_t KECCAK_D_SEQUENCE[25] = {
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4
};
static constexpr uint64_t KECCAK_ROUND_CONSTANTS[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

Keccak::Keccak(KeccakParam param) {
    bool initialised = false;
    switch (param) {
        case KeccakParam::SHA3_224:
            initialised = Init(448, 0x06);
            break;
        case KeccakParam::SHA3_256:
            initialised = Init(512, 0x06);
            break;
        case KeccakParam::SHA3_384:
            initialised = Init(768, 0x06);
            break;
        case KeccakParam::SHA3_512:
            initialised = Init(1024, 0x06);
            break;
        case KeccakParam::SHAKE128:
            initialised = Init(256, 0x1F);
            break;
        case KeccakParam::SHAKE256:
            initialised = Init(512, 0x1F);
            break;
    }
    if (!initialised)
        throw std::invalid_argument("Could not initialize Keccak state");
}

bool Keccak::Init(const size_t capacity, const uint8_t delimited_suffix) {
    const size_t rate = 1600 - capacity;
    if (rate + capacity != 1600 || rate % 8 !=0 )
        return false;
    rate_ = rate;
    capacity_ = capacity;
    rate_in_bytes_ = rate_ / 8;
    state_.fill(0);
    delimited_suffix_ = delimited_suffix;
    pos_ = 0;
    finalized_ = false;
    return true;
}

void Keccak::Absorb(std::span<const std::byte> data) {
    if (finalized_)
        return;

    size_t input_offset = 0;
    while (input_offset < data.size()) {
        size_t space = rate_in_bytes_ - pos_;
        size_t block_size = (std::min)(space, data.size() - input_offset);
        for (size_t i = 0; i < block_size; i++) {
            state_[pos_ + i] ^= static_cast<uint8_t>(data[input_offset + i]);
        }

        pos_ += block_size;
        input_offset += block_size;

        if (pos_ == rate_in_bytes_) {
            XOF_F1600(state_);
            pos_ = 0;
        }
    }
}

bool Keccak::Squeeze(std::span<uint8_t> output) {
    if (!finalized_)
        XOF_FINALIZE();
    size_t output_index = 0;
    size_t output_len = output.size();
    while (output_len > 0) {
        size_t block_size = (std::min)(rate_in_bytes_ - pos_, output_len);

        std::memcpy(&output[output_index], &state_[pos_], block_size);

        pos_ += block_size;
        output_index += block_size;
        output_len -= block_size;

        if (pos_ == rate_in_bytes_) {
            XOF_F1600(state_);
            pos_ = 0;
        }
    }
    return true;
}

void Keccak::XOF_F1600_ON_LANES(std::array<uint64_t, 25> &lanes) {
    for (int round = 0; round < 24; round++) {
        uint64_t C[5], D[5];
        for (int x = 0; x < 5; x++) {
            C[x] = lanes[(x * 5) + 0] ^ lanes[(x * 5) + 1] ^ lanes[(x * 5) + 2] ^ lanes[(x * 5) + 3] ^ lanes[(x * 5) + 4];
        }
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ ((C[(x + 1) % 5] << 1) | (C[(x + 1) % 5] >> 63));
        }
        for (int i = 0; i < 25; i++) {
            lanes[i] ^= D[KECCAK_D_SEQUENCE[i]];
        }

        uint64_t current_bit = lanes[5];
        for (int t = 0; t < 24; t++) {
            uint64_t new_bit = lanes[KECCAK_PI_INDEX[t]];
            lanes[KECCAK_PI_INDEX[t]] = (current_bit << KECCAK_RHO_OFFSETS[t]) | (current_bit >> (64 - KECCAK_RHO_OFFSETS[t]));
            current_bit = new_bit;
        }

        for (int y = 0; y < 5; y++) {
            uint64_t b0 = lanes[y];
            uint64_t b1 = lanes[5 + y];
            uint64_t b2 = lanes[10 + y];
            uint64_t b3 = lanes[15 + y];
            uint64_t b4 = lanes[20 + y];

            lanes[y] = (b0 ^ (~b1) & b2);
            lanes[y + 5] = (b1 ^ (~b2) & b3);
            lanes[y + 10] = b2 ^ ((~b3) & b4);
            lanes[y + 15] = b3 ^ ((~b4) & b0);
            lanes[y + 20] = b4 ^ ((~b0) & b1);
        }
        lanes[0] ^= KECCAK_ROUND_CONSTANTS[round];
    }

}

void Keccak::XOF_F1600(std::array<uint8_t, 200> &state) {
    std::array<uint64_t,25> lanes;
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 5; y++) {
            std::memcpy(&lanes[x * 5 + y], &state[8 * (x + 5 * y)], sizeof(uint64_t));
        }
    }

    XOF_F1600_ON_LANES(lanes);

    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 5; y++) {
            std::memcpy(&state[8 * (x + 5 * y)], &lanes[x * 5 + y], sizeof(uint64_t));
        }
    }
}

void Keccak::XOF_FINALIZE() {
    if (finalized_)
        return;
    state_[pos_] ^= delimited_suffix_;
    if ((delimited_suffix_ & 0x80) && (pos_ == rate_in_bytes_ -1)) {
        XOF_F1600(state_);
    }
    state_[rate_in_bytes_ - 1] ^= 0x80;
    XOF_F1600(state_);
    pos_ = 0;
    finalized_ = true;
}
