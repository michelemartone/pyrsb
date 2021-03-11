import math
import sys
import numpy
import scipy
from scipy.sparse import csr_matrix
import pyrsb
from pyrsb import rsb_matrix
from pyrsb import _print_vec, rsb_time, _err_check, rsb_dtype, _dt2dt
prv_t = rsb_dtype

def gen_dense(dtype,order,nrhs,n=1000):
    nr = n
    nc = nr
    d = numpy.ones(shape=(nr,nc), dtype=dtype)
    x = numpy.ones([nc, nrhs], dtype=dtype, order=order)
    y = numpy.ones([nr, nrhs], dtype=dtype, order=order)
    return [y,d,x]

dtypes = [numpy.float64, numpy.float32, numpy.complex128, numpy.complex64]

def from_dense(format,d):
        if format == 'rsb':
            return rsb_matrix(d)
        else:
            return csr_matrix(d)

class BenchMatvecs():
    params = [ dtypes, [1, 2, 4, 8], ['C','F'], ['csr','rsb'] ]
    param_names = ["dtype","nrhs","order","format"]

    def setup(self,dtype,nrhs,order,format):
        [self.y,self.d,self.x] = gen_dense(dtype,order,nrhs,n=1000)
        self.a = from_dense(format,self.d)

    def do_bare_mul(self,format):
        if format == 'rsb':
            self.a._spmm(self.x,self.y)
        else:
            scipy.sparse._sparsetools.csr_matvecs(self.a.shape[0], self.a.shape[1], self.x.shape[1], self.a.indptr, self.a.indices, self.a.data, self.x.ravel(), self.y.ravel())

    def time_bare_mul(self,dtype,nrhs,order,format):
        self.do_bare_mul(format)

    def time_mul(self,dtype,nrhs,order,format):
        y = self.a * self.x

    def peakmem_bare_mul(self,dtype,nrhs,order,format):
        self.do_bare_mul(format)

    def peakmem_mul(self,dtype,nrhs,order,format):
        y = self.a * self.x

class BenchDenseMatvecs(BenchMatvecs):
    params = [ dtypes, [1, 2, 4, 8], ['C','F'], ['csr','rsb'] ]
    param_names = ["dtype","nrhs","order","format"]

    def setup(self,dtype,nrhs,order,format):
        [self.y,self.d,self.x] = gen_dense(dtype,order,nrhs,n=2000)
        self.a = from_dense(format,self.d)

class BenchDenseCtor():
    params = [ dtypes, ['csr','rsb'] ]
    param_names = ["dtype","format"]

    def setup(self,dtype,format):
        [self.y,self.d,self.x] = gen_dense(dtype,'C',1)
        self.a = from_dense(format,self.d)

    def time_ctor(self,dtype,format):
        a = from_dense(format,self.d)

    def peakmem_ctor(self,dtype,format):
        a = from_dense(format,self.d)
