#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "hamming_index.hpp"

namespace py = pybind11;

namespace {

using U64Array = py::array_t<std::uint64_t>;

std::string shape_string(const py::array& array) {
    std::string shape = "(";
    for (py::ssize_t i = 0; i < array.ndim(); ++i) {
        if (i > 0) {
            shape += ", ";
        }
        shape += std::to_string(array.shape(i));
    }
    if (array.ndim() == 1) {
        shape += ",";
    }
    shape += ")";
    return shape;
}

U64Array require_u64_array(const py::array& array, const char* name) {
    if (!py::isinstance<U64Array>(array)) {
        throw py::type_error(std::string(name) + " must have dtype uint64.");
    }

    auto typed = py::reinterpret_borrow<U64Array>(array);

    if (!(typed.flags() & py::array::c_style)) {
        throw py::value_error(
            std::string(name) + " must be C-contiguous."
        );
    }

    if (reinterpret_cast<std::uintptr_t>(typed.data()) %
            alignof(std::uint64_t) != 0) {
        throw py::value_error(
            std::string(name) + " must be uint64-aligned."
        );
    }

    return typed;
}

std::unique_ptr<lsh::HammingIndex> make_index(
    const py::array& signatures
) {
    auto words = require_u64_array(signatures, "signatures");

    if (words.ndim() != 2 ||
        words.shape(1) !=
            static_cast<py::ssize_t>(lsh::kWordsPerSignature)) {
        throw py::value_error(
            "signatures must have shape (N, 4); got " +
            shape_string(words) + "."
        );
    }

    if (words.shape(0) == 0) {
        throw py::value_error("signatures cannot be empty.");
    }

    return std::make_unique<lsh::HammingIndex>(
        words.data(),
        static_cast<std::size_t>(words.shape(0))
    );
}

py::tuple search(
    const lsh::HammingIndex& index,
    const py::array& query,
    py::ssize_t k
) {
    if (k < 1) {
        throw py::value_error("k must be at least 1.");
    }

    auto words = require_u64_array(query, "query");

    if (words.ndim() != 1 ||
        words.shape(0) !=
            static_cast<py::ssize_t>(lsh::kWordsPerSignature)) {
        throw py::value_error(
            "query must have shape (4,); got " +
            shape_string(words) + "."
        );
    }

    std::array<std::uint64_t, lsh::kWordsPerSignature> query_copy{};
    std::copy_n(words.data(), lsh::kWordsPerSignature, query_copy.data());

    lsh::SearchResult result;
    {
        py::gil_scoped_release release;
        result = index.search(
            query_copy.data(),
            static_cast<std::size_t>(k)
        );
    }

    const auto count = static_cast<py::ssize_t>(result.indices.size());
    py::array_t<std::int64_t> indices(count);
    py::array_t<std::uint16_t> distances(count);

    std::copy(
        result.indices.begin(),
        result.indices.end(),
        indices.mutable_data()
    );
    std::copy(
        result.distances.begin(),
        result.distances.end(),
        distances.mutable_data()
    );

    return py::make_tuple(indices, distances);
}

}  

PYBIND11_MODULE(_lsh_cpp, module) {
    module.doc() = "C++ Hamming search over 256-bit signatures.";

    py::class_<lsh::HammingIndex>(module, "HammingIndex")
        .def(py::init(&make_index), py::arg("signatures"))
        .def("search", &search, py::arg("query"), py::arg("k"))
        .def("__len__", &lsh::HammingIndex::size);
}