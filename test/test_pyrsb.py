"""
Test file for the PyRSB interface to LIBRSB.
Note that this file tests the PyRSB API, not LIBRSB itself.
"""

import numpy
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from rsb import rsb_matrix
from rsb import _print_vec, rsb_time, _err_check
import pytest
from pytest import raises as assert_raises
from time import sleep


def test__err_check_ok():
    _err_check(0)
    _err_check(0,want_strict=True)


def test__err_check_err():
    _err_check(1,want_strict=False)
    with assert_raises(AssertionError):
        _err_check(1,want_strict=True)


def gen_tri_csr():
    V = [11., 12., 22.]
    J = (0,1,1)
    P = (0,2,3)
    return [V,J,P,2,2,3]


def gen_tri():
    V = [11.0, 12.0, 22.0]
    I = [0, 0, 1]
    J = [0, 1, 1]
    return [V,I,J,2,2,3]


def gen_x(n,nrhs=1,order='C'):
    x = numpy.empty([n, nrhs], dtype=scipy.double, order=order)
    for i in range(nrhs):
        print(i)
        x[:, i] = i+1
    return x


def test__print_vec():
    for order in ['C', 'F']:
        x = gen_x(3,nrhs=2,order=order)
        assert 0 == _print_vec(x)


def test__print_vec_throw():
    with assert_raises(ValueError):
        _print_vec(numpy.empty([1 ], dtype=scipy.double))


def test_init_from_none():
    mat = rsb_matrix(None)
    assert mat.shape == (0, 0)
    assert mat.nnz == 0
    assert mat._is_unsymmetric() == True
    assert mat.dtype == numpy.float64
    assert mat.ndim == 2
    assert mat.has_sorted_indices == False


def test_init_from_none_dtype_D():
    mat = rsb_matrix(None,dtype='d')


def test_init_from_dims_dtype_D():
    mat = rsb_matrix([1,1],dtype='d')


def test_init_from_none_dtype_Wrong():
    with assert_raises(TypeError):
        mat = rsb_matrix(None,dtype='W')


def test_init_from_dims_dtype_Wrong():
    with assert_raises(TypeError):
        mat = rsb_matrix([1,1],dtype='W')


def test_init_from_none_none():
    mat = rsb_matrix(None,None)
    assert mat.shape == (0, 0)
    assert mat.nnz == 0
    assert mat._is_unsymmetric() == True


def test_init_tuple_csr():
    [V,J,P,nr,nc,nnz] = gen_tri_csr()
    mat = rsb_matrix((V, J, P),[nr,nc])
    assert mat.nnz == nnz
    assert mat.shape == (nr, nc)
    assert mat._is_unsymmetric() == True


def test_init_tuples():
    [V,I,J,nr,nc,nnz] = gen_tri()
    mat = rsb_matrix((V, (I, J)))
    assert mat.shape == (nr, nc)
    assert mat.nnz == nnz
    assert mat._is_unsymmetric() == True


def test_init_tuples_and_dims():
    [V,I,J,nr,nc,nnz] = gen_tri()
    mat = rsb_matrix((V, (I, J)),[nr,nc])
    assert mat.shape == (nr, nc)
    assert mat.nnz == nnz
    assert mat._is_unsymmetric() == True


def test_init_tuples__raises():
    [V,I,J,nr,nc,nnz] = gen_tri()
    with assert_raises(TypeError):
        mat = rsb_matrix((V, (I)))


def test_init_tuples_and_dims_raises():
    [V,I,J,nr,nc,nnz] = gen_tri()
    with assert_raises(TypeError):
        mat = rsb_matrix((V, (I)),[3,3])


def test_init_tuples_fixed_1():
    [V,I,J,nr,nc,nnz] = gen_tri()
    mat = rsb_matrix([V, (I)])
    with assert_raises(AssertionError):
         assert ( mat.nnz == 0 )


def test_init_tuples_fixed_2():
    [V,J,P,nr,nc,nnz] = gen_tri_csr()
    mat = rsb_matrix((V, J, P),[nr,nc])
    assert mat.nnz == nnz
    assert mat.shape == (nr, nc)
    assert mat._is_unsymmetric() == True


def test_init_tuple_to_fix_3():
    # TODO: shall align rsb_matrix to scipy's
    mat = rsb_matrix(([1.,1.], ([-1,-2], [-1,-2])))
    assert mat.shape == (0, 0)
    assert mat.nnz == 0
    assert mat._is_unsymmetric() == True

def test_init_tuple_except():
    with assert_raises(AssertionError):
        mat = rsb_matrix(([1.,1.], ([0,1], [0,1])),[2,-2])

def test__refresh():
    [V,I,J,nr,nc,nnz] = gen_tri();
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    rmat._refresh()

def test_init_from_csc():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csc_matrix((V, (I, J)),[nr,nc])
    rmat = rsb_matrix(cmat)
    assert ((cmat - rmat.tocsr())).nnz == 0


def test_init_from_dense():
    d = numpy.ones(shape=(2,2), dtype=numpy.float64)
    cmat = csr_matrix(d)
    rmat = rsb_matrix(cmat)
    assert ((cmat - rmat.tocsr())).nnz == 0


def test_do_print():
    [V,I,J,nr,nc,nnz] = gen_tri();
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    rmat.do_print(brief=True)
    rmat.do_print(brief=False)


def test_nonzero():
    [V,I,J,nr,nc,nnz] = gen_tri();
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    cmat = csr_matrix((V, (I, J)),[nr,nc])
    [cI,cJ] = cmat.nonzero();
    [rI,rJ] = rmat.nonzero();
    # order matters: won't work for any matrix
    assert ( cI == rI ).all()
    assert ( cJ == rJ ).all()


def test_io_bytes_ctor():
    [sV,sI,sJ,nr,nc,nnz] = gen_tri();
    smat = rsb_matrix((sV, (sI, sJ)))
    filename = b"pyrsb_test.tmp.mtx"
    smat.save(filename)
    lmat = rsb_matrix(filename)
    [lI,lJ,lV] = lmat.find();
    assert ( sV == lV ).all()
    assert ( sI == lI ).all()
    assert ( sJ == lJ ).all()


def test_sleep():
    t0 = rsb_time()
    sleep(0.001)
    t1 = rsb_time()
    assert (t1 > t0)


def test_todense():
    [V,I,J,nr,nc,nnz] = gen_tri();
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    cmat = csr_matrix((V, (I, J)),[nr,nc])
    assert ( rmat.todense() == cmat.todense() ).all()


def test_tocsr():
    [V,I,J,nr,nc,nnz] = gen_tri();
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    cmat = csr_matrix((V, (I, J)),[nr,nc])
    assert ((cmat - rmat.tocsr())).nnz == 0


def test_find():
    [V,I,J,nr,nc,nnz] = gen_tri();
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    cmat = csr_matrix((V, (I, J)),[nr,nc])
    [cI,cJ,cV] = scipy.sparse.find(cmat);
    [rI,rJ,rV] = rmat.find();
    # order matters: won't work for any matrix
    assert ( cV == rV ).all()
    assert ( cI == rI ).all()
    assert ( cJ == rJ ).all()


def test_tril():
    [V,I,J,nr,nc,nnz] = gen_tri();
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    cmat = csr_matrix((V, (I, J)),[nr,nc])
    [cI,cJ,cV] = scipy.sparse.find(scipy.sparse.tril(cmat));
    [rI,rJ,rV] = rmat.tril();
    # order matters: won't work for any matrix
    assert ( cV == rV ).all()
    assert ( cI == rI ).all()
    assert ( cJ == rJ ).all()


def test_triu():
    [V,I,J,nr,nc,nnz] = gen_tri();
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    cmat = csr_matrix((V, (I, J)),[nr,nc])
    [cI,cJ,cV] = scipy.sparse.find(scipy.sparse.triu(cmat));
    [rI,rJ,rV] = rmat.triu();
    # order matters: won't work for any matrix
    assert ( cV == rV ).all()
    assert ( cI == rI ).all()
    assert ( cJ == rJ ).all()


def test_mini_self_print_test():
    """Call mini self test."""
    [V,I,J,nr,nc,nnz] = gen_tri()
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    rmat.mini_self_print_test()


def test__find_block():
    [V,I,J,nr,nc,nnz] = gen_tri()
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    rmat._find_block(0,rmat.nr()-1,0,rmat.nc()-1)
    [rI,rJ,rV] = rmat.find();
    # order matters: won't work for any matrix
    assert ( V == rV ).all()
    assert ( I == rI ).all()
    assert ( J == rJ ).all()


def test_init_tuples_sym():
    [V,I,J,nr,nc,nnz] = gen_tri();
    mat = rsb_matrix((V, (I, J)),sym="S")
    assert mat.shape == (nr, nc)
    assert mat.nnz == nnz
    assert mat._is_unsymmetric() == False


def test_init_tuples_wrong_sym():
    [V,I,J,nr,nc,nnz] = gen_tri();
    with assert_raises(ValueError):
        mat = rsb_matrix((V, (I, J)),sym="W")


def test_spmv__mul__():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    x = gen_x(nc)
    assert ( (rmat * x) == (cmat * x) ).all()

def test_spmv_1D_N():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    nrhs = 1
    for order in ['C', 'F']:
        x = gen_x(nc,nrhs,order)
        y = numpy.empty([nr, nrhs], dtype=scipy.double, order=order)
        y[:, :] = 0.0
        x = x[:,0]
        y = y[:,0]
        rmat._spmv(x,y)
        assert ( y == (cmat * x) ).all()

def test_autotune_simple():
    [V,I,J,nr,nc,nnz] = gen_tri();
    omat = rsb_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    rmat.autotune()
    assert( rmat.todense() == omat.todense() ).all()

def test__spmul():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    assert( (cmat*cmat).todense() == (rmat*rmat).todense() ).all()

def test__spadd():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    assert( (cmat+cmat).todense() == (rmat+rmat).todense() ).all()

def test_spmv_1D_T():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    nrhs = 1
    for order in ['C', 'F']:
        for transA in ['T', b'T', ord('T')]:
            x = gen_x(nr,nrhs,order)
            y = numpy.empty([nc, nrhs], dtype=scipy.double, order=order)
            y[:, :] = 0.0
            x = x[:,0]
            y = y[:,0]
            rmat._spmv(x,y,transA=transA)
            assert ( y == ( cmat.transpose() * x) ).all()

def test_spmm_C():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    assert rmat.shape == cmat.shape
    assert rmat.nnz == cmat.nnz
    nrhs = 2
    order = 'C'
    x = gen_x(nc,nrhs,order)
    y = numpy.empty([nr, nrhs], dtype=scipy.double, order=order)
    y[:, :] = 0.0
    rmat._spmm(x,y)
    assert ( y == (cmat * x) ).all()

def test_spmm_wrong_transA():
    [V,I,J,nr,nc,nnz] = gen_tri();
    rmat = rsb_matrix((V, (I, J)))
    nrhs = 2
    x = gen_x(nc,nrhs)
    y = numpy.empty([nr, nrhs], dtype=scipy.double)
    y[:, :] = 0.0
    with assert_raises(ValueError):
       rmat._spmm(x,y,transA='?')

def test_spmm_C_T():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    assert rmat.shape == cmat.shape
    assert rmat.nnz == cmat.nnz
    nrhs = 2
    order = 'C'
    x = gen_x(nc,nrhs,order)
    y = numpy.empty([nr, nrhs], dtype=scipy.double, order=order)
    y[:, :] = 0.0
    rmat._spmm(x,y,transA=b'T')
    assert ( y == (cmat.transpose() * x) ).all()

def test_spmm_C_T_forms():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    assert rmat.shape == cmat.shape
    assert rmat.nnz == cmat.nnz
    nrhs = 2
    order = 'C'
    x = gen_x(nc,nrhs,order)
    y = numpy.empty([nr, nrhs], dtype=scipy.double, order=order)
    for transA in ['T', b'T', ord('T')]:
        y[:, :] = 0.0
        rmat._spmm(x,y,transA=transA)
        assert ( y == (cmat.transpose() * x) ).all()

def test_spmm_F():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    assert rmat.shape == cmat.shape
    assert rmat.nnz == cmat.nnz
    nrhs = 2
    order='F'
    x = gen_x(nr,nrhs,order)
    y = numpy.empty([nc, nrhs], dtype=scipy.double, order=order)
    y[:, :] = 0.0
    rmat._spmm(x,y)
    assert ( y == (cmat * x) ).all()

def test_spmm_F_T():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    assert rmat.shape == cmat.shape
    assert rmat.nnz == cmat.nnz
    nrhs = 2
    order = 'F'
    x = gen_x(nr,nrhs,order)
    y = numpy.zeros([nc, nrhs], dtype=scipy.double, order=order)
    rmat._spmm(x,y,transA=b'T')
    assert ( y == (cmat.transpose() * x) ).all()

def test_spmm_permitted_mismatch():
    [V,I,J,nr,nc,nnz] = gen_tri();
    rmat = rsb_matrix((V, (I, J)))
    nrhs = 1
    x1 = gen_x(nc,nrhs,order='F')
    x2 = gen_x(nc,nrhs,order='C')
    assert ( (rmat * x1).shape == (rmat * x2).shape )
    assert ( (rmat * x1) == (rmat * x2) ).all()

def test_spmm__mul__():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    assert rmat.shape == cmat.shape
    assert rmat.nnz == cmat.nnz
    nrhs = 2
    x = gen_x(nc,nrhs)
    assert ( (rmat * x) == (cmat * x) ).all()

def test_rescaled():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J))).rescaled(2.0)
    x = gen_x(nc)
    assert ( (rmat * x) == (2.0 * cmat * x) ).all()

def test_demo():
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
    x = numpy.empty([nc, nrhs], dtype=a.dtype, order=order)
    y = numpy.empty([nr, nrhs], dtype=a.dtype, order=order)
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
