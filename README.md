
# PyRSB

[librsb](http://librsb.sourceforge.net/) is a **high performance sparse matrix
library** implementing the Recursive Sparse Blocks format,
which is especially well suited for
multiplications in **iterative methods** on **huge symmetric sparse matrices**.

**PyRSB is a Cython-based Python interface to librsb.**

So far, PyRSB is only a prototype: **prospective users and collaborators feedback are sought**; [please contact me](http://librsb.sourceforge.net/#a_contacts) to feedback and help.

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
  Make sure you have `cython`, `scipy`, `numpy`. installed.
- If you want the `Makefile` to build librsb (in this directory):
 `make all-local` will attempt downloading librsb-1.2.0-rc7 from the 
 web and building it here before building pyrsb.
 If the file is in place, it won't download it a second time.
 After that, `make local-librsb-pyrsb` (or `make lp`) will build pyrsb
 using local librsb, then run it.
- `make test` will test benchmark code using `test.py` (*to compare speed to SciPy*)
- `make b` will also produce graphs (requires `gnuplot`)

## Example Usage

```python
import numpy
import scipy
from scipy.sparse import csr_matrix
from rsb          import rsb_matrix
V=[11.,12.,22.]
I=[  0,  0,  1]
J=[  0,  1,  1]
c=csr_matrix((V,(I,J)))
print(c)
# several constructor forms, as with csr_matrix:
a=rsb_matrix((V,(I,J)))
a=rsb_matrix((V,(I,J)),[3,3])
a=rsb_matrix((V, I,J))
a=rsb_matrix((V, I,J),sym='S') # symmetric example
print(a)
a=rsb_matrix(          (4,4))
a=rsb_matrix(c)
nrhs=1 # set to nrhs>1 to multiply by multiple vectors at once
nr=a.shape()[0]
nc=a.shape()[1]
x=numpy.empty([nc,nrhs],dtype=scipy.double)
y=numpy.empty([nr,nrhs],dtype=scipy.double)
x[:,:]=1.0
y[:,:]=0.0
print(a)
print(x)
print(y)
import rsb # import operators
# a.autotune() # makes only sense for large matrices
# matrix-vector multiply
y=y+a*x; # equivalent to y=y+c*x
print(y)
del a
```

## License
GPLv3+
