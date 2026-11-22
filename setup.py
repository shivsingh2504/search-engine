from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


setup(
    name="semantic-search-lsh",
    packages=[],
    ext_modules=[
        Pybind11Extension(
            "src.semantic_search._lsh_cpp",
            ["cpp/hamming_index.cpp", "cpp/bindings.cpp"],
            include_dirs=["cpp"],
            cxx_std=20,
        ),
    ],
    cmdclass={"build_ext": build_ext},
)