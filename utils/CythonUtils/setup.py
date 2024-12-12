from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


# Define the extension module
extensions = [
    Extension("cy_argwhere", ["cy_argwhere.pyx"],
              include_dirs=[numpy.get_include()]),  # Add NumPy include dir here
]

setup(
    name="CythonUtils",
    ext_modules=cythonize(extensions),
)