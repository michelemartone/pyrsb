import numpy
import scipy
from scipy.sparse import csr_matrix
from rsb import rsb_matrix


def gen_tri():
    V = [11.0, 12.0, 22.0]
    I = [0, 0, 1]
    J = [0, 1, 1]
    return [V,I,J,2,2,3]


def test_init_tuple():
    [V,I,J,nr,nc,nnz] = gen_tri();
    mat = rsb_matrix((V, I, J))
    assert mat.shape == (nr, nc)
    assert mat.nnz() == nnz
    assert mat._is_unsymmetric() == True


def test_init_tuples():
    [V,I,J,nr,nc,nnz] = gen_tri();
    mat = rsb_matrix((V, (I, J)))
    assert mat.shape == (nr, nc)
    assert mat.nnz() == nnz
    assert mat._is_unsymmetric() == True


def test_init_tuples_and_dims():
    [V,I,J,nr,nc,nnz] = gen_tri();
    mat = rsb_matrix((V, (I, J)),[nr,nc])
    assert mat.shape == (nr, nc)
    assert mat.nnz() == nnz
    assert mat._is_unsymmetric() == True


def test_init_tuples_sym():
    [V,I,J,nr,nc,nnz] = gen_tri();
    mat = rsb_matrix((V, (I, J)),sym="S")
    assert mat.shape == (nr, nc)
    assert mat.nnz() == nnz
    assert mat._is_unsymmetric() == False


def test_spmv():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    assert rmat.shape == cmat.shape
    assert rmat.nnz() == cmat.nnz
    nrhs = 1
    x = numpy.empty([nc, nrhs], dtype=scipy.double)
    x[:, :] = 1.0
    assert ( (rmat * x) == (cmat * x) ).all()

def test_spmm():
    [V,I,J,nr,nc,nnz] = gen_tri();
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    assert rmat.shape == cmat.shape
    assert rmat.nnz() == cmat.nnz
    nrhs = 2
    x = numpy.empty([nc, nrhs], dtype=scipy.double)
    x[:, :] = 1.0
    assert ( (rmat * x) == (cmat * x) ).all()

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
