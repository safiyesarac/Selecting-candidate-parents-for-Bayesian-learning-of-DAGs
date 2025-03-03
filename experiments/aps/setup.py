from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "/home/gulce/Downloads/thesis/experiments/aps/_aps.pyx",
        compiler_directives={'language_level': 3},
        cplus=True
    )
)
