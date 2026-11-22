#include "hamming_index.hpp"

#include <bits/stdc++.h>
#include <stdexcept>
#include <bit>
#include <utility>

namespace lsh{
  namespace{
    std::uint16_t hamming_dist(
      const std::uint64_t* left,
      const std::uint64_t* right
    ){
      return static_cast<std::uint64_t>(
        std:popcount(left[0] ^ right[0])+
        std:popcount(left[1] ^ right[1])+
        std:popcount(left[2] ^ right[2])+
        std:popcount(left[3] ^ right[3])+
      );
    }
  }

}