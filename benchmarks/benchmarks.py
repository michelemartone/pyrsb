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

class BenchDense():
    params = [ dtypes, ['csr','rsb'] ]
    param_names = ["dtype","format"]
    def setup(self,dtype,format):
        [self.y,self.d,self.x] = gen_dense(dtype)
        if format == 'rsb':
            self.a = rsb_matrix(self.d)
        else:
            self.a = csr_matrix(self.d)

    def time_bare(self,dtype,format):
        if format == 'rsb':
            self.a._spmm(self.x,self.y)
        else:
            scipy.sparse._sparsetools.csr_matvecs(self.a.shape[0], self.a.shape[1], self.x.shape[1], self.a.indptr, self.a.indices, self.a.data, self.x.ravel(), self.y.ravel())

    def time_rsb(self,dtype,format):
        y = self.a * self.x

