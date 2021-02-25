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
a = rsb_matrix((V, (I, J)))

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
