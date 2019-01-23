from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name = 'rsb',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("rsb", ["rsb.pyx"], libraries=["rsb","z","hwloc","gfortran"])] # FIXME
)
