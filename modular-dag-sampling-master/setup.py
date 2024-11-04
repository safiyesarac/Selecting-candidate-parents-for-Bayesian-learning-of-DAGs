from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

extensions = [
    Extension(
        "nonsymmetric_sampler",
        ["nonsymmetric_sampler.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"],
        include_dirs=[
            os.path.join(current_dir, 'src'),  # Add src/ directory to include_dirs
        ],
    )
]

setup(
    name="nonsymmetric_sampler",
    ext_modules=cythonize(extensions),
)
