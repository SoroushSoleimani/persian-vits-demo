import os
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

_DIR = Path(__file__).resolve().parent
os.chdir(str(_DIR))

if os.name == "nt":
    compile_args = ["/O2", "/openmp"]
    link_args = []
else:
    compile_args = ["-O3", "-fopenmp"]
    link_args = ["-fopenmp"]

ext = Extension(
    name="core",
    sources=["core.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="monotonic_align",
    ext_modules=cythonize([ext], language_level="3"),
)
