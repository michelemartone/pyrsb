"""setuptools installer script for PyRSB."""
import os
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from time import gmtime, strftime

if True:
    VERSION = "0.2.20210303"
else:
    if os.environ.get("PYRSB_VERSION"):
        VERSION = os.environ.get("PYRSB_VERSION")
    else:
        VERSION = strftime("0.2.%Y%m%d%H%M%S", gmtime())

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

from numpy import get_include

stream = os.popen("librsb-config --libdir")
lib_dir = stream.read().strip()

stream = os.popen("librsb-config --I_opts")
inc_dir = stream.read().strip()[2:]

INCLUDE_DIRS = [get_include(), inc_dir]
LIB_DIRS = [lib_dir]

setup(
    name="pyrsb",
    version=VERSION,
    author="Michele Martone",
    author_email="michelemartone@users.sourceforge.net",
    description="PyRSB: a Cython-based Python interface to librsb",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/michelemartone/pyrsb",
    py_modules=['pyrsb'],
    project_urls={
        "Bug Tracker": "https://github.com/michelemartone/pyrsb/issues",
        "Source Code": "https://github.com/michelemartone/pyrsb",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        # "Operating System :: POSIX :: Linux",
    ],
    ext_modules=[
        Extension(
            "rsb",
            ["rsb.pyx", "librsb.pxd"],
            libraries=["rsb", "z", "hwloc", "gfortran"],
            library_dirs=LIB_DIRS,
            include_dirs=INCLUDE_DIRS,
        )
    ],
    setup_requires=["numpy", "scipy"],
    install_requires=["numpy", "scipy"],
    cmdclass={"build_ext": build_ext},
    # package_data = { '': ['rsb.pyx','*.py'] },
    include_package_data=True,
    python_requires=">=3.7",
)
