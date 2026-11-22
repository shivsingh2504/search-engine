#include "hamming_index.hpp"

#include <bits/stdc++.h>
#include <stdexcept>
#include <bit>
#include <utility>

namespace lsh {
namespace {

std::uint16_t hamming_distance(
    const std::uint64_t* left,
    const std::uint64_t* right
) {
    return static_cast<std::uint16_t>(
        std::popcount(left[0] ^ right[0]) +
        std::popcount(left[1] ^ right[1]) +
        std::popcount(left[2] ^ right[2]) +
        std::popcount(left[3] ^ right[3])
    );
}

}  

HammingIndex::HammingIndex(const std::uint64_t* words, std::size_t count)
    : count_(count) {
    if (words == nullptr) {
        throw std::invalid_argument("Signature data cannot be null.");
    }
    if (count == 0) {
        throw std::invalid_argument("The index needs at least one signature.");
    }
    if (count > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max())) {
        throw std::length_error("Too many signatures.");
    }

    words_.assign(words, words + count * kWordsPerSignature);
}

SearchResult HammingIndex::search(
    const std::uint64_t* query,
    std::size_t k
) const {
    if (query == nullptr) {
        throw std::invalid_argument("Query signature cannot be null.");
    }
    if (k == 0) {
        throw std::invalid_argument("k must be at least 1.");
    }

    k = std::min(k, count_);

    
    using Entry = std::pair<std::uint16_t, std::int64_t>;
    std::vector<Entry> storage;
    storage.reserve(k);
    std::priority_queue<Entry> heap(std::less<Entry>{}, std::move(storage));

    for (std::size_t row = 0; row < count_; ++row) {
        const auto* signature =
            words_.data() + row * kWordsPerSignature;

        Entry candidate{
            hamming_distance(query, signature),
            static_cast<std::int64_t>(row)
        };

        if (heap.size() < k) {
            heap.push(candidate);
        } else if (candidate < heap.top()) {
            heap.pop();
            heap.push(candidate);
        }
    }

    SearchResult result;
    result.indices.resize(heap.size());
    result.distances.resize(heap.size());

    
    for (std::size_t i = heap.size(); i-- > 0;) {
        result.distances[i] = heap.top().first;
        result.indices[i] = heap.top().second;
        heap.pop();
    }

    return result;
}

}  