from rsb import rsb_matrix


def test_init_tuple():
    V = [11.0, 12.0, 22.0]
    I = [0, 0, 1]
    J = [0, 1, 1]
    mat = rsb_matrix((V, I, J))
    assert mat.shape == (2, 2)
    assert mat.nnz() == 3
    assert mat._is_unsymmetric() == True


def test_init_tuples():
    V = [11.0, 12.0, 22.0]
    I = [0, 0, 1]
    J = [0, 1, 1]
    mat = rsb_matrix((V, (I, J)))
    assert mat.shape == (2, 2)
    assert mat.nnz() == 3
    assert mat._is_unsymmetric() == True


def test_init_tuples_and_dims():
    V = [11.0, 12.0, 22.0]
    I = [0, 0, 1]
    J = [0, 1, 1]
    mat = rsb_matrix((V, (I, J)),[3,3])
    assert mat.shape == (3, 3)
    assert mat.nnz() == 3
    assert mat._is_unsymmetric() == True


def test_init_tuples_sym():
    V = [11.0, 12.0, 22.0]
    I = [0, 0, 1]
    J = [0, 1, 1]
    mat = rsb_matrix((V, (I, J)),sym="S")
    assert mat.shape == (2, 2)
    assert mat.nnz() == 3
    assert mat._is_unsymmetric() == False


def test_demo():
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
