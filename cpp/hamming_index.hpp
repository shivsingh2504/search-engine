#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lsh {

inline constexpr std::size_t kWordsPerSignature = 4;

struct SearchResult {
    std::vector<std::int64_t> indices;
    std::vector<std::uint16_t> distances;
};

class HammingIndex {
public:
    HammingIndex(const std::uint64_t* words, std::size_t count);

    std::size_t size() const noexcept { return count_; }

    SearchResult search(const std::uint64_t* query, std::size_t k) const;

private:
    std::vector<std::uint64_t> words_;
    std::size_t count_;
};

}  // namespace lsh
