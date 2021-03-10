import math
import sys
import numpy
import scipy
from scipy.sparse import csr_matrix
import pyrsb
from pyrsb import rsb_matrix
from pyrsb import _print_vec, rsb_time, _err_check, rsb_dtype, _dt2dt
prv_t = rsb_dtype

def gen_dense(dtype):
    nr = 1000
    nc = nr
    nrhs = 4
    d = numpy.ones(shape=(nr,nc), dtype=dtype)
    order = "C"
    x = numpy.ones([nc, nrhs], dtype=dtype, order=order)
    y = numpy.ones([nr, nrhs], dtype=dtype, order=order)
    return [y,d,x]

dtypes = [numpy.float64, numpy.float32]
class BenchCsr():
    params = dtypes
    param_names = ["dtype"]
    def setup(self,dtype):
        [self.y,self.d,self.x] = gen_dense(dtype)
        self.c = csr_matrix(self.d)

    def time_csr(self,dtype):
        y = self.c * self.x

    def time_csr_bare(self,dtype):
        scipy.sparse._sparsetools.csr_matvecs(self.c.shape[0], self.c.shape[1], self.x.shape[1], self.c.indptr, self.c.indices, self.c.data, self.x.ravel(), self.y.ravel())

class BenchRsb():
    params = dtypes
    param_names = ["dtype"]
    def setup(self,dtype):
        [self.y,self.d,self.x] = gen_dense(dtype)
        self.a = rsb_matrix(self.d)

    def time_rsb_bare(self,dtype):
        self.a._spmm(self.x,self.y)

    def time_rsb(self,dtype):
        y = self.a * self.x

