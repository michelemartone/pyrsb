"""
Test file for the PyRSB interface to LIBRSB.
Note that this file tests the PyRSB API, not LIBRSB itself.
"""

import numpy
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from pyrsb import rsb_matrix
from pyrsb import _get_rsb_threads, _print_vec, rsb_time, _err_check, rsb_dtype, _dt2dt
import pytest
from pytest import raises as assert_raises
from time import sleep


rsb_real_dtypes = [ numpy.float32, numpy.float64 ]
rsb_cplx_dtypes = [ numpy.complex64, numpy.complex128 ]
rsb_dtypes = rsb_real_dtypes + rsb_cplx_dtypes
prv_t = rsb_dtype
max_idx_bpnz = 8


def test__err_check_ok():
    _err_check(0)
    _err_check(0,want_strict=True)


def test__err_check_err():
    _err_check(1,want_strict=False)
    with assert_raises(AssertionError):
        _err_check(1,want_strict=True)


def test__get_rsb_threads():
    assert _get_rsb_threads() > 0


def gen_tri_csr_larger():
    V = [11.,11.]
    J = [0,9]
    P = [0,1,1,1,1,1,1,1,1,1,2]
    return [V,J,P,10,10,2]


def gen_tri_csr(dtype=prv_t):
    V = numpy.array([11., 12., 22.],dtype=dtype)
    J = (0,1,1)
    P = (0,2,3)
    return [V,J,P,2,2,3]


def gen_tri(dtype=prv_t):
    V = numpy.array([11.0, 12.0, 22.0],dtype=dtype)
    I = [0, 0, 1]
    J = [0, 1, 1]
    return [V,I,J,2,2,3]


def gen_rect(dtype=prv_t):
    V = numpy.array([11.0, 60.0],dtype=dtype)
    I = [0, 5]
    J = [0, 0]
    return [V,I,J,6,1,2]


@pytest.fixture(params=rsb_dtypes)
def f_gen_rect(request):
    return gen_rect(dtype=request.param)


@pytest.fixture(params=rsb_dtypes)
def f_gen_tri(request):
    return gen_tri(dtype=request.param)


@pytest.fixture(params=rsb_cplx_dtypes)
def f_gen_tri_complex(request):
    return gen_tri(dtype=request.param)


def gen_x(n,nrhs=1,order='C', dtype=prv_t):
    x = numpy.empty([n, nrhs], dtype=dtype, order=order)
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
        _print_vec(numpy.empty([1 ], dtype=prv_t))


def test_init_from_none():
    mat = rsb_matrix(None)
    assert mat.shape == (0, 0)
    assert mat.nnz == 0
    assert mat._is_unsymmetric() == True
    assert mat.ndim == 2
    assert mat.has_sorted_indices == False


def test_init_from_none_dtype_D():
    mat = rsb_matrix(None,dtype='d')
    assert mat.shape == (0, 0)
    assert mat.nnz == 0
    assert mat._is_unsymmetric() == True
    assert mat.dtype == _dt2dt('d')
    assert mat.ndim == 2
    assert mat.has_sorted_indices == False


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
    assert mat._get_symchar() == 'G'
    #TODO; enable these one 1.2.0.10 and 1.3 available:
    #assert mat._idx_bpnz() > 0
    #assert mat._idx_bpnz() <= max_idx_bpnz
    assert mat._total_size > 0


def test_init_tuple_csr_f32():
    [V,J,P,nr,nc,nnz] = gen_tri_csr(dtype=numpy.float32)
    mat = rsb_matrix((V, J, P),[nr,nc])
    assert mat.nnz == nnz
    assert mat.shape == (nr, nc)
    assert mat._is_unsymmetric() == True
    assert mat._get_symchar() == 'G'


def test_init_tuple_csr():
    [V,J,P,nr,nc,nnz] = gen_tri_csr()
    mat = rsb_matrix((V, J, P),[nr,nc])
    assert mat.nnz == nnz
    assert mat.shape == (nr, nc)
    assert mat._is_unsymmetric() == True
    assert mat._get_symchar() == 'G'


def test_init_tuple_csr_larger():
    [V,J,P,nr,nc,nnz] = gen_tri_csr_larger()
    mat = rsb_matrix((V, J, P),[nr,nc])
    assert mat.nnz == nnz
    assert mat.shape == (nr, nc)
    assert mat._is_unsymmetric() == True


def test_init_tuple_csr_smaller_JA_err():
    [V,J,P,nr,nc,nnz] = gen_tri_csr_larger()
    with assert_raises(AssertionError):
        mat = rsb_matrix((V + [1.], J, P),[nr,nc])


def test_init_tuple_csr_larger_IP_err():
    [V,J,P,nr,nc,nnz] = gen_tri_csr_larger()
    with assert_raises(AssertionError):
        mat = rsb_matrix((V, J, P + [nnz]),[nr,nc])


def test_init_tuples_err(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    with assert_raises(AssertionError):
        mat = rsb_matrix((V, (I+[0], J)))


def test_init_tuples(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    mat = rsb_matrix((V, (I, J)))
    assert mat.shape == (nr, nc)
    assert mat.nnz == nnz
    assert mat._is_unsymmetric() == True
    assert mat._get_typechar() in [ 'S', 'D', 'C', 'Z' ]
    assert mat._idx_bpnz() > 0
    assert mat._idx_bpnz() <= max_idx_bpnz


def test_init_tuples_and_dims(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    mat = rsb_matrix((V, (I, J)),[nr,nc])
    assert mat.shape == (nr, nc)
    assert mat.nnz == nnz
    assert mat._is_unsymmetric() == True
    assert mat._idx_bpnz() > 0
    assert mat._idx_bpnz() <= max_idx_bpnz


def test_init_tuples__raises(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    with assert_raises(TypeError):
        mat = rsb_matrix((V, (I)))


def test_init_tuples_and_dims_raises(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    with assert_raises(TypeError):
        mat = rsb_matrix((V, (I)),[3,3])


def test_init_tuples_fixed_1(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    mat = rsb_matrix([V, (I)])
    with assert_raises(AssertionError):
         assert ( mat.nnz == 0 )


def test_init_tuples_fixed_2():
    [V,J,P,nr,nc,nnz] = gen_tri_csr()
    mat = rsb_matrix((V, J, P),[nr,nc])
    assert mat.nnz == nnz
    assert mat.shape == (nr, nc)
    assert mat._is_unsymmetric() == True


def test_init_tuple_fixed_3():
    with assert_raises(ValueError):
        mat = rsb_matrix(([1.,1.], ([-1,-2], [-1,-2])))


def test_init_tuple_except():
    with assert_raises(AssertionError):
        mat = rsb_matrix(([1.,1.], ([0,1], [0,1])),[2,-2])


def test__refresh(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    rmat._refresh()


def test_init_from_csc(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    cmat = csc_matrix((V, (I, J)),[nr,nc])
    rmat = rsb_matrix(cmat)
    assert ((cmat - rmat.tocsr())).nnz == 0


def test_init_from_dense():
    d = numpy.ones(shape=(2,2), dtype=prv_t)
    cmat = csr_matrix(d)
    rmat = rsb_matrix(cmat)
    assert ((cmat - rmat.tocsr())).nnz == 0


def test_do_print(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    rmat.do_print(brief=True)
    rmat.do_print(brief=False)


def test_nonzero(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    cmat = csr_matrix((V, (I, J)),[nr,nc])
    [cI,cJ] = cmat.nonzero();
    [rI,rJ] = rmat.nonzero();
    # order matters: won't work for any matrix
    assert ( cI == rI ).all()
    assert ( cJ == rJ ).all()


def test_render_to_stdout():
    mat = rsb_matrix([1,1],dtype='d')
    mat.render()


# def test_render_to_stdout():
#     mat = rsb_matrix([1,1],dtype='d')
#     mat.render("one.eps")


def test_io_bytes_ctor(f_gen_tri):
    for dtype in rsb_dtypes:
        [sV,sI,sJ,nr,nc,nnz] = f_gen_tri
        smat = rsb_matrix((sV, (sI, sJ)))
        filename = b"pyrsb_test.tmp.mtx"
        smat.save(filename)
        lmat = rsb_matrix(filename,dtype=dtype)
        [lI,lJ,lV] = lmat.find();
        assert ( sV == lV ).all()
        assert ( sI == lI ).all()
        assert ( sJ == lJ ).all()


def test_sleep():
    t0 = rsb_time()
    sleep(0.001)
    t1 = rsb_time()
    assert (t1 > t0)


@pytest.fixture(params=rsb_dtypes)
def f_gen_mats(request):
    [V,I,J,nr,nc,nnz] = gen_tri(dtype=request.param)
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    cmat = csr_matrix((V, (I, J)),[nr,nc])
    return [rmat,cmat]


def test_todense(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    assert ( rmat.todense() == cmat.todense() ).all()


def test_tocsr(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    assert ((cmat - rmat.tocsr())).nnz == 0


def test_find(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    [cI,cJ,cV] = scipy.sparse.find(cmat);
    [rI,rJ,rV] = rmat.find();
    # order matters: won't work for any matrix
    assert ( cV == rV ).all()
    assert ( cI == rI ).all()
    assert ( cJ == rJ ).all()


def test_tril(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    [cI,cJ,cV] = scipy.sparse.find(scipy.sparse.tril(cmat));
    [rI,rJ,rV] = rmat.tril();
    # order matters: won't work for any matrix
    assert ( cV == rV ).all()
    assert ( cI == rI ).all()
    assert ( cJ == rJ ).all()


def test_triu(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    [cI,cJ,cV] = scipy.sparse.find(scipy.sparse.triu(cmat));
    [rI,rJ,rV] = rmat.triu();
    # order matters: won't work for any matrix
    assert ( cV == rV ).all()
    assert ( cI == rI ).all()
    assert ( cJ == rJ ).all()


def test_conj(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    pytest.skip("Not implemented yet", allow_module_level=True)
    assert ( rmat.conj() == cmat.conj() )


def test_conjugate(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    pytest.skip("Not implemented yet", allow_module_level=True)
    assert ( rmat.conjugate() == cmat.conjugate() )


def test_mini_self_print_test(f_gen_tri):
    """Call mini self test."""
    [V,I,J,nr,nc,nnz] = f_gen_tri
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    rmat._mini_self_print_test()


def test__find_block(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    rmat._find_block(0,rmat.nr()-1,0,rmat.nc()-1)
    [rI,rJ,rV] = rmat.find();
    # order matters: won't work for any matrix
    assert ( V == rV ).all()
    assert ( I == rI ).all()
    assert ( J == rJ ).all()


def test__otn2obc_rect_n(f_gen_rect):
    [V,I,J,nr,nc,nnz] = f_gen_rect
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    nrhs = 2
    (cm_o,cm_ldB,cm_ldC) = rmat._otn2obc(False,'N',nrhs)
    assert ( (cm_ldB,cm_ldC) == ( nc, nr ) )
    (rm_o,rm_ldB,rm_ldC) = rmat._otn2obc(True ,'N',nrhs)
    assert ( (rm_ldB,rm_ldC) == ( nrhs, nrhs ) )


def test__otn2obc_rect_t(f_gen_rect):
    [V,I,J,nr,nc,nnz] = f_gen_rect
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    nrhs = 2
    (cm_o,cm_ldB,cm_ldC) = rmat._otn2obc(False,'T',nrhs)
    assert ( (cm_ldB,cm_ldC) == ( nr, nc ) )
    (rm_o,rm_ldB,rm_ldC) = rmat._otn2obc(True ,'T',nrhs)
    assert ( (rm_ldB,rm_ldC) == ( nrhs, nrhs ) )


def test__otn2obc_tri(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    rmat = rsb_matrix((V, (I, J)),[nr,nc])
    nrhs = 1
    (cm_o,cm_ldB,cm_ldC) = rmat._otn2obc(False,'N',nrhs)
    assert ( (cm_ldB,cm_ldC) == ( nr, nc ) )
    (rm_o,rm_ldB,rm_ldC) = rmat._otn2obc(True ,'N',nrhs)
    assert ( (rm_ldB,rm_ldC) == ( nrhs, nrhs ) )


def test_init_tuples_sym(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    mat = rsb_matrix((V, (I, J)),sym="S")
    assert mat.shape == (nr, nc)
    assert mat.nnz == nnz
    assert mat._is_unsymmetric() == False
    assert mat._get_symchar() == 'S'


def test_init_tuples_herm(f_gen_tri_complex):
    [V,I,J,nr,nc,nnz] = f_gen_tri_complex
    mat = rsb_matrix((V, (I, J)),sym="H",dtype=V.dtype) # Note: dtype not inherited from V.
    assert mat.shape == (nr, nc)
    assert mat.nnz == nnz
    assert mat._is_unsymmetric() == False
    assert mat._get_symchar() == 'H'
    assert mat._is_complex()


def test_init_tuples_wrong_sym(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    with assert_raises(ValueError):
        mat = rsb_matrix((V, (I, J)),sym="W")


def test_spmv__mul__(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    x = gen_x(rmat.nc())
    assert ( (rmat * x) == (cmat * x) ).all()


def test_spmv_matvec(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    x = gen_x(rmat.nc())
    assert ( (rmat.matvec(x) ) == (cmat * x) ).all()


def test_spmv_matvec_gmres():
     for dtype in [float]:
        A = csr_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=dtype)
        A = rsb_matrix(A)
        n = A.shape[0]
        b = numpy.sin(numpy.array(range(1,n+1)))
        from scipy.sparse.linalg import gmres
        x, exitCode = gmres(A, b)
        assert ( exitCode == 0 )


def test_spmv_1D_N(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    nrhs = 1
    for order in ['C', 'F']:
        x = gen_x(rmat.nc(),nrhs,order)
        y = numpy.empty([rmat.nr(), nrhs], dtype=prv_t, order=order)
        y[:, :] = 0.0
        x = x[:,0]
        y = y[:,0]
        rmat._spmv(x,y)
        assert ( y == (cmat * x) ).all()


def test_spmv_1D_N_alpha(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    nrhs = 1
    for order in ['C', 'F']:
        x = gen_x(rmat.nc(),nrhs,order)
        y = numpy.empty([rmat.nr(), nrhs], dtype=prv_t, order=order)
        y[:, :] = 0.0
        x = x[:,0]
        y = y[:,0]
        rmat._spmv(x,y,alpha=2)
        assert ( y == 2 * (cmat * x) ).all()


def test_autotune_simple(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    omat = rsb_matrix((V, (I, J)))
    cmat = omat.copy()
    assert( omat.todense() == cmat.todense() ).all()


def test_autotune_simple(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    omat = rsb_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)))
    rmat.autotune()
    assert( rmat.todense() == omat.todense() ).all()


def test__spmul():
    for dtype in rsb_dtypes:
        [V,I,J,nr,nc,nnz] = gen_tri(dtype=dtype);
        cmat = csr_matrix((V, (I, J)))
        rmat = rsb_matrix((V, (I, J)))
        assert( (cmat*cmat).todense() == (rmat*rmat).todense() ).all()


def test__spadd(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    assert( (cmat+cmat).todense() == (rmat+rmat).todense() ).all()


def test_spmv_1D_T(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    nrhs = 1
    for order in ['C', 'F']:
        for transA in ['T', b'T', ord('T')]:
            x = gen_x(rmat.nr(),nrhs,order)
            y = numpy.empty([rmat.nc(), nrhs], dtype=prv_t, order=order)
            y[:, :] = 0.0
            x = x[:,0]
            y = y[:,0]
            rmat._spmv(x,y,transA=transA)
            assert ( y == ( cmat.transpose() * x) ).all()


def test_spmm_C(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    assert rmat.shape == cmat.shape
    assert rmat.nnz == cmat.nnz
    nrhs = 2
    order = 'C'
    x = gen_x(rmat.nc(),nrhs,order)
    y = numpy.empty([rmat.nr(), nrhs], dtype=prv_t, order=order)
    y[:, :] = 0.0
    rmat._spmm(x,y)
    assert ( y == (cmat * x) ).all()


def test_spmm_wrong_transA(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    rmat = rsb_matrix((V, (I, J)))
    nrhs = 2
    x = gen_x(nc,nrhs)
    y = numpy.empty([nr, nrhs], dtype=prv_t)
    y[:, :] = 0.0
    with assert_raises(ValueError):
       rmat._spmm(x,y,transA='?')


def test_spmm_C_T(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    assert rmat.shape == cmat.shape
    assert rmat.nnz == cmat.nnz
    nrhs = 2
    order = 'C'
    x = gen_x(rmat.nc(),nrhs,order)
    y = numpy.empty([rmat.nr(), nrhs], dtype=prv_t, order=order)
    y[:, :] = 0.0
    rmat._spmm(x,y,transA='T')
    assert ( y == (cmat.transpose() * x) ).all()


def test_spmm_C_T_forms(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    assert rmat.shape == cmat.shape
    assert rmat.nnz == cmat.nnz
    nrhs = 2
    order = 'C'
    x = gen_x(rmat.nc(),nrhs,order)
    y = numpy.empty([rmat.nr(), nrhs], dtype=prv_t, order=order)
    for transA in ['T', b'T', ord('T')]:
        y[:, :] = 0.0
        rmat._spmm(x,y,transA=transA)
        assert ( y == (cmat.transpose() * x) ).all()


def test_spmm_F(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    assert rmat.shape == cmat.shape
    assert rmat.nnz == cmat.nnz
    nrhs = 2
    order='F'
    x = gen_x(rmat.nr(),nrhs,order)
    y = numpy.empty([rmat.nc(), nrhs], dtype=prv_t, order=order)
    y[:, :] = 0.0
    rmat._spmm(x,y)
    assert ( y == (cmat * x) ).all()


def test_spmm_F_T(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    assert rmat.shape == cmat.shape
    assert rmat.nnz == cmat.nnz
    nrhs = 2
    order = 'F'
    x = gen_x(rmat.nr(),nrhs,order)
    y = numpy.zeros([rmat.nc(), nrhs], dtype=prv_t, order=order)
    rmat._spmm(x,y,transA='T')
    assert ( y == (cmat.transpose() * x) ).all()


def test_spmm_permitted_mismatch(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    rmat = rsb_matrix((V, (I, J)))
    nrhs = 1
    x1 = gen_x(nc,nrhs,order='F')
    x2 = gen_x(nc,nrhs,order='C')
    assert ( (rmat * x1).shape == (rmat * x2).shape )
    assert ( (rmat * x1) == (rmat * x2) ).all()


def test_spmm__mul__(f_gen_mats):
    [rmat,cmat] = f_gen_mats
    assert rmat.shape == cmat.shape
    assert rmat.nnz == cmat.nnz
    nrhs = 2
    x = gen_x(rmat.nc(),nrhs)
    assert ( (rmat * x) == (cmat * x) ).all()


def test_rescaled(f_gen_tri):
    [V,I,J,nr,nc,nnz] = f_gen_tri
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J))).rescaled(2.0)
    x = gen_x(nc)
    assert ( (rmat * x) == (2.0 * cmat * x) ).all()


def test_rescaled_f64():
    [V,I,J,nr,nc,nnz] = gen_tri(dtype=numpy.float64);
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)),dtype=numpy.float64)
    rmat = rsb_matrix((V, (I, J)),dtype=numpy.float64).rescaled(2.0)
    x = gen_x(nc, dtype=numpy.float64)
    rmat.save()
    assert ( (rmat * x) == (2.0 * cmat * x) ).all()


def test_rescaled_c64():
    [V,I,J,nr,nc,nnz] = gen_tri(dtype=numpy.complex128);
    cmat = csr_matrix((V, (I, J)))
    rmat = rsb_matrix((V, (I, J)),dtype=numpy.complex128).rescaled(2.0)
    x = gen_x(nc, dtype=numpy.complex128)
    assert ( (rmat * x) == (2.0 * cmat * x) ).all()


def test_rescaled_any_type():
    for dtype in rsb_dtypes:
        [V,I,J,nr,nc,nnz] = gen_tri(dtype=dtype);
        cmat = csr_matrix((V, (I, J)))
        rmat = rsb_matrix((V, (I, J)),dtype=dtype).rescaled(2.0)
        x = gen_x(nc, dtype=dtype)
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
    a = rsb_matrix((V, (I, J)), sym="S")  # symmetric example
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
    # import pyrsb # import operators
    # a.autotune() # makes only sense for large matrices
    y = y + a * x
    # equivalent to y=y+c*x
    print(y)
    del a
