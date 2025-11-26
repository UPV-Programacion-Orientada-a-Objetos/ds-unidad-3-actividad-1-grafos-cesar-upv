from setuptools import Extension, setup
from Cython.Build import cythonize

# script de compilacion hibrida
extensions = [
    Extension(
        name="neuronet_core",
        sources=[
            "src/cython/grafocore.pyx",
            "src/cpp/GrafoDisperso.cpp",
        ],
        include_dirs=["src/cpp"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )
]

setup(
    name="neuronet-core",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    zip_safe=False,
)
