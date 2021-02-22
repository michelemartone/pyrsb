
# PyRSB

[![GPL enforced badge](https://img.shields.io/badge/GPL-enforced-blue.svg "This project enforces the GPL.")](https://gplenforced.org)
[![Build Status](https://travis-ci.org/michelemartone/pyrsb.svg?branch=master)](https://travis-ci.org/michelemartone/pyrsb)

[librsb](http://librsb.sourceforge.net/) is a **high performance sparse matrix
library** implementing the Recursive Sparse Blocks format,
which is especially well suited for
multiplications in **iterative methods** on **huge symmetric sparse matrices**.

**PyRSB is a Cython-based Python interface to librsb.**

On multicore machines, PyRSB can be several times faster than e.g. `scipy.sparse.csr_matrix()`.
For an example how to invoke it with no overhead, [see the advanced example](#ExampleAdvancedUsage).

So far, PyRSB is a prototype tested on Linux only.
The librsb library offers much more, and can make PyRSB much more powerful.
**Prospective users and collaborators feedback are sought**; [please contact me](http://librsb.sourceforge.net/#a_contacts) to feedback and help.

## Features

The following functionality is implemented:

  * Initialization with `rsb.rsb_matrix()` styled as [`scipy.sparse.csr_matrix()`](https://docs.scipy.org/doc/scipy/reference/sparse.html).
  * Conversion from `scipy.sparse.csr_matrix()`.
  * Multiplication by vector/multivector.
  * Rows/columns through `nr=a.shape()[0]`/`nr=a.shape()[1]`, or `nr()`/`nc()`.
  * `find()`, `find_block()`, `tril()`, `triu()`, `shape()`, `nnz()`.
  * `print`'able.
  * PyRSB-Specific: `autotune()`, `do_print()`.
  * load from a Matrix Market file, e.g. `rsb.rsb_file_mtx_load(bytes(filename,encoding='utf-8'))`

## Build and Use

- If you have librsb installed:
 `make` shall build and test.
- Make sure you have `cython`, `scipy`, `numpy`. installed.
- If you want to install librsb on Ubuntu or Debian:
 `sudo apt-get install librsb-dev` shall suffice.
  Other operating systems have librsb, too.
  Please check yours.
  Or check [librsb](http://librsb.sourceforge.net/)'s web site.
- If you want the `Makefile` to build librsb (in this directory):
 `make all-local` will attempt downloading librsb-1.2.0.9 from the
 web and building it here before building pyrsb.
 If the file is in place, it won't download it a second time.
 After that, `make local-librsb-pyrsb` (or `make lp`) will build pyrsb
 using local librsb, then run it.
- `make test` will test benchmark code using `test.py` (*to compare speed to SciPy*)
- `make b` will also produce graphs (requires `gnuplot`)

## Example Usage

```python
# Example: demo1.py
"""
pyrsb demo
"""

import numpy
import scipy
from scipy.sparse import csr_matrix
from rsb import rsb_matrix

V = [11.0, 12.0, 22.0]
I = [0, 0, 1]
J = [0, 1, 1]
c = csr_matrix((V, (I, J)))
print(c)
# several constructor forms, as with csr_matrix:
a = rsb_matrix((V, (I, J)))
a = rsb_matrix((V, (I, J)), [3, 3])
a = rsb_matrix((V, I, J))
a = rsb_matrix((V, I, J), sym="S")  # symmetric example
print(a)
a = rsb_matrix((4, 4))
a = rsb_matrix(c)
nrhs = 1  # set to nrhs>1 to multiply by multiple vectors at once
nr = a.shape[0]
nc = a.shape[1]
order = "F"
x = numpy.empty([nc, nrhs], dtype=scipy.double, order=order)
y = numpy.empty([nr, nrhs], dtype=scipy.double, order=order)
x[:, :] = 1.0
y[:, :] = 0.0
print(a)
print(x)
print(y)
# import rsb # import operators
# a.autotune() # makes only sense for large matrices
y = y + a * x
# equivalent to y=y+c*x
print(y)
del a
```

## <a id="ExampleAdvancedUsage"></a>Example Advanced Usage ##

```python
# Example: demo2.py
"""
pyrsb demo
"""

import numpy
import scipy
from scipy.sparse import csr_matrix
from rsb import rsb_matrix

V = [11.0, 12.0, 22.0]
I = [0, 0, 1]
J = [0, 1, 1]
a = rsb_matrix((V, I, J))

nrhs = 4  # set to nrhs>1 to multiply by multiple vectors at once
nr = a.shape[0]
nc = a.shape[1]

# Choose Fortran or "by columns" order here.
order = "F"
x = numpy.empty([nc, nrhs], dtype=scipy.double, order=order)
y = numpy.empty([nr, nrhs], dtype=scipy.double, order=order)
x[:, :] = 1.0
y[:, :] = 0.0
print(a)
print(x)
print(y)

# Autotuning example: use it if you need many multiplication iterations on huge matrices (>>1e6 nonzeroes).
# Here general (nrhs=1) case:
a.autotune()
# Here with all the autotuning parameters specified:
a.autotune(1.0,0,1,2.0,ord('N'),1.0,nrhs,ord('F'),1.0,False)

# Inefficient: reallocate y
y = y + a * x
# Inefficient: reallocate y
y += a * x

# Equivalent but more efficient: don't reallocate y
a._spmm(x,y)

print(y)

del a
```

## License
GPLv3+
