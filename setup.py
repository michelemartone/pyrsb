from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext_modules = [
        Extension("rsb", ["rsb.pyx"],
            libraries=["rsb","z","hwloc"])]
setup(
    name = 'rsb',
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext_modules)
)
