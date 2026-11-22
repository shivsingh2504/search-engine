#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "hamming_index.hpp"

namespace py = pybind11;
namespace{
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
  U64Array require_u64_array(const py::array& array,const char* name){
    if(!py::isinstance<U64Array>(array)){
      throw py::type_error(std::string(name) + " must have dtype uint64.");
    }
    auto typed = py::reinterpret_borrow<U64Array>(array);
    if(!(typed.flags() & py::array::c_style)){
      throw py::value_error(std::string(name) + " must be C-contiguous.");
    }
    return typed;
  }
  

}