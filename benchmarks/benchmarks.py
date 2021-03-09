import math
import sys
import numpy
import scipy
from scipy.sparse import csr_matrix
import pyrsb
from pyrsb import rsb_matrix
from pyrsb import _print_vec, rsb_time, _err_check, rsb_dtype, _dt2dt
prv_t = rsb_dtype

class Dense:
    """
    Sample PyRSB benchmark, from dense.
    """
    def __init__(self):
        self.nr = 1000
        self.nc = self.nr
        self.nrhs = 4
        self.d = numpy.ones(shape=(self.nr,self.nc), dtype=prv_t)
        order = "C"
        self.x = numpy.ones([self.nc, self.nrhs], dtype=rsb_dtype, order=order)
        self.y = numpy.ones([self.nr, self.nrhs], dtype=rsb_dtype, order=order)

class BenchCsr(Dense):
    def setup(self):
        self.c = csr_matrix(self.d)

    def time_csr(self):
        y = self.c * self.x

    def time_csr_bare(self):
        scipy.sparse._sparsetools.csr_matvecs(self.c.shape[0], self.c.shape[1], self.x.shape[1], self.c.indptr, self.c.indices, self.c.data, self.x.ravel(), self.y.ravel())

class BenchRsb(Dense):
    def setup(self):
        self.a = rsb_matrix(self.d)

    def time_rsb_bare(self):
        self.a._spmm(self.x,self.y)

    def time_rsb(self):
        y = self.a * self.x


