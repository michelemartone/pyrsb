# cython: language_level=3
"""
Recursive Sparse Blocks matrix format.
librsb interface for Python.
Proof of concept, limited interface code, aims at compatibility with scipy.sparse.
Author: Michele Martone
License: GPLv3+
"""

cimport librsb as lr

import numpy as np
cimport numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

import scipy as sp

import cython
cimport cython

__all__ = [
    'rsb_matrix', 'rsb_time', 'rsb_dtype', '_get_rsb_threads',
    '_print_vec', '_err_check', '_dt2dt'
]

verbose=0

ctypedef fused any_t:
    cython.doublecomplex
    cython.floatcomplex
    cython.double
    cython.float

#rsb_dtype = np.complex64
#ctypedef float complex prv_t

#rsb_dtype = np.complex128
#ctypedef double complex prv_t

rsb_dtype = np.float64
ctypedef double prv_t

#rsb_dtype = np.float32
#ctypedef float prv_t

def _is_complex_rsb_supported(dtype):
    return ( dtype == np.complex128 or dtype == np.complex64 )

def _dt2dt(dtype):
    if isinstance(dtype, np.dtype) or isinstance(dtype, type):
        if dtype == np.float64:
            return np.float64
        elif dtype == np.float32:
            return np.float32
        elif dtype == np.complex128:
            return np.complex128
        elif dtype == np.complex64:
            return np.complex64
    else:
        if dtype.upper() == 'D':
            return np.float64
        elif dtype.upper() == 'S':
            return np.float32
        elif dtype.upper() == 'Z':
            return np.complex128
        elif dtype.upper() == 'C':
            return np.complex64
    raise TypeError("Wrong data type: ", dtype)

def _dt2tc(dtype):
    if isinstance(dtype, np.dtype) or isinstance(dtype, type):
        if dtype == np.float64:
            return lr.RSB_NUMERICAL_TYPE_DOUBLE
        elif dtype == np.float32:
            return lr.RSB_NUMERICAL_TYPE_FLOAT
        elif dtype == np.complex128:
            return lr.RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX
        elif dtype == np.complex64:
            return lr.RSB_NUMERICAL_TYPE_FLOAT_COMPLEX
    else:
        if dtype.upper() == 'D':
            return lr.RSB_NUMERICAL_TYPE_DOUBLE
        elif dtype.upper() == 'S':
            return lr.RSB_NUMERICAL_TYPE_FLOAT
        elif dtype.upper() == 'Z':
            return lr.RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX
        elif dtype.upper() == 'C':
            return lr.RSB_NUMERICAL_TYPE_FLOAT_COMPLEX
    raise TypeError("Wrong data type: ", dtype)

cpdef rsb_lib_init():
    """Initialize librsb."""
    if verbose:
        print("Initializing librsb")
    cdef lr.rsb_err_t errval = lr.rsb_lib_init(NULL)
    _err_check(errval,want_strict=True)
    return errval

cpdef rsb_lib_exit():
    """Finalize librsb."""
    if verbose:
        print("Finalizing librsb")
    cdef lr.rsb_err_t errval = lr.rsb_lib_exit(NULL)
    _err_check(errval,want_strict=True)
    return errval

cpdef rsb_time():
    """Return current time."""
    cdef lr.rsb_time_t rt
    rt = <lr.rsb_time_t>lr.rsb_time()
    return rt

cpdef _get_rsb_threads():
    """Return current LIBRSB threads."""
    cdef int iof = lr.RSB_IO_WANT_EXECUTING_THREADS # FIXME
    #cdef lr.rsb_opt_t iof = lr.RSB_IO_WANT_EXECUTING_THREADS
    cdef lr.rsb_int_t nt = 0
    cdef lr.rsb_err_t errval = lr.rsb_lib_get_opt(iof, &nt)
    _err_check(errval)
    return nt

def _print_vec(np.ndarray x, mylen=0):
    """Print a vector, possibly overriding its length (which is DANGEROUS)."""
    cdef lr.rsb_coo_idx_t ylv = 0
    if x.ndim != 2:
        raise ValueError
    ylv = len(x)
    if mylen is not 0:
        ylv = mylen
    return lr.rsb_file_vec_save(NULL, _dt2tc(x.dtype), <lr.cvoid_ptr>x.data, ylv)

def _err_check(lr.rsb_err_t errval,want_strict=False):
    """
    Basic error checking.
    (specific to rsb).
    """
    cdef size_t buflen = 256
    cdef char buf[256]
    if ( errval ):
        lr.rsb_strerror_r(errval,buf,buflen)
        errval = lr.RSB_ERR_NO_ERROR
        print("Error reported by librsb: ", str(buf,'ascii'))
        if want_strict:
            assert False
        return False
    return True

cdef class rsb_matrix:
    """
    Recursive Sparse Blocks matrix
    """
    cdef lr.rsb_mtx_ptr mtxAp
    cdef lr.rsb_type_t typecode 
    cdef lr.rsb_coo_idx_t ncA
    cdef lr.rsb_coo_idx_t nrA
    cdef lr.rsb_nnz_idx_t nnzA # see http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.nnz.html#scipy.sparse.csr_matrix.nnz
    cdef lr.rsb_blk_idx_t nsubmA
    cdef lr.rsb_real_t idx_bpnz
    cdef size_t total_size
    cdef lr.rsb_flags_t flagsA
    cdef type dtypeA
    idx_dtype = np.int32
    ndim = 2
    format = 'rsb'

    def _get_dtype(self):
        return self.dtypeA

    def _get_typechar(self):
        return chr(_dt2tc(self._get_dtype()))

    def _get_symchar(self):
        cdef lr.rsb_err_t errval
        cdef lr.rsb_flags_t flagsA = lr.RSB_FLAG_NOFLAGS
        errval = lr.rsb_mtx_get_info(self.mtxAp, lr.RSB_MIF_MATRIX_FLAGS__TO__RSB_FLAGS_T,&flagsA)
        if ( ( flagsA & (lr.RSB_FLAG_HERMITIAN | lr.RSB_FLAG_SYMMETRIC ) ) == lr.RSB_FLAG_NOFLAGS ):
            return 'G'
        elif ( ( flagsA & (lr.RSB_FLAG_HERMITIAN) ) == lr.RSB_FLAG_HERMITIAN):
            return 'H'
        else:
            return 'S'

    def _psf2lsf(self, sym):
        """
        Python Symmetry Flag to librsb Symmetry Flag.
        """
        if sym == 'U' or sym == ord('U') or sym == b'U':
                return lr.RSB_FLAG_NOFLAGS
        if sym == 'S' or sym == ord('S') or sym == b'S':
                return lr.RSB_FLAG_LOWER_SYMMETRIC
        if sym == 'H' or sym == ord('H') or sym == b'H':
                return lr.RSB_FLAG_LOWER_HERMITIAN
        raise ValueError("Unrecognized symmetry")

    def _prt2lt(self, transA):
        """
        Python RSB transA to librsb transA.
        """
        if transA == 'N' or transA == ord('N') or transA == b'N':
                return lr.RSB_TRANSPOSITION_N
        if transA == 'T' or transA == ord('T') or transA == b'T':
                return lr.RSB_TRANSPOSITION_T
        if transA == 'C' or transA == ord('C') or transA == b'C':
                return lr.RSB_TRANSPOSITION_C
        raise ValueError("Unrecognized transA")

    def _spmm(self,np.ndarray[any_t, ndim=2] x, np.ndarray[any_t, ndim=2] y, transA='N', alpha = 1.0, beta = 1.0):
        """
        Sparse Matrix by matrix product based on rsb_spmm().
        See scipy.sparse.spmatrix._mul_multivector().
        (specific to rsb).
        """
        cdef lr.rsb_coo_idx_t nrhs = x.shape[1]
        cdef lr.rsb_nnz_idx_t ldB, ldC
        cdef lr.rsb_trans_t transA_ = self._prt2lt(transA)
        cdef lr.rsb_flags_t lr_order = lr.RSB_FLAG_NOFLAGS
        cdef lr.rsb_err_t errval
        corder =  x.flags.c_contiguous
        (lr_order,ldB,ldC)=self._otn2obc(corder,transA,nrhs)
        cdef np.ndarray talpha = np.array([alpha],dtype=self.dtype)
        cdef np.ndarray tbeta = np.array([beta],dtype=self.dtype)
        assert x.flags.c_contiguous == y.flags.c_contiguous
        if nrhs is not 1:
            assert lr_order==lr.RSB_FLAG_WANT_COLUMN_MAJOR_ORDER or lr_order==lr.RSB_FLAG_WANT_ROW_MAJOR_ORDER
        if x.shape[1] is not y.shape[1]:
           errval = lr.RSB_ERR_BADARGS
        else:
           errval = lr.rsb_spmm(transA_, talpha.data, self.mtxAp, nrhs, lr_order, <lr.cvoid_ptr>x.data, ldB, tbeta.data, <lr.void_ptr>y.data, ldC);
        _err_check(errval)
        return errval

    def _spmv(self,np.ndarray[any_t, ndim=1] x, np.ndarray[any_t, ndim=1] y, transA='N', alpha = 1.0, beta = 1.0):
        """
        Sparse Matrix by vector product based on rsb_spmv().
        See scipy.sparse.spmatrix._mul_vector().
        (specific to rsb).
        """
        cdef lr.rsb_coo_idx_t incX = 1, incY = 1
        cdef lr.rsb_trans_t transA_ = self._prt2lt(transA)
        cdef lr.rsb_err_t errval
        cdef np.ndarray talpha = np.array([alpha],dtype=self.dtype)
        cdef np.ndarray tbeta = np.array([beta],dtype=self.dtype)
        errval = lr.rsb_spmv(transA_, talpha.data, self.mtxAp, <lr.cvoid_ptr>x.data, incX, tbeta.data, <lr.void_ptr>y.data, incY)
        _err_check(errval)
        return errval

    def __init__(self,arg1=None,shape=None,sym='U',dtype=None):
        cdef lr.rsb_err_t errval
        cdef np.ndarray VA
        cdef np.ndarray IP # IA/PA
        cdef np.ndarray JA
        self.nrA=0
        self.ncA=0
        cdef lr.rsb_blk_idx_t brA = 0, bcA = 0
        cdef lr.rsb_flags_t flagsA = lr.RSB_FLAG_NOFLAGS
        self.flagsA = flagsA
        self.mtxAp = NULL
        self.nnzA=0
        V = None
        I = None
        J = None
        P = None
        self.dtypeA = None

        if dtype is not None:
            self.dtypeA = _dt2dt(dtype)

        if arg1 is not None:
            if isinstance(arg1, str):
                arg1 = bytes(arg1,encoding="utf-8")
            if isinstance(arg1, bytes):
                if dtype is None:
                    self.dtypeA = _dt2dt(rsb_dtype)
                self.typecode = _dt2tc(self.dtype)
                self.mtxAp = lr.rsb_file_mtx_load(arg1,flagsA,self.typecode,&errval)
                _err_check(errval)
                self._refresh()
                return
            elif type(arg1) == type(self):
                self = arg1.copy()
                return
            elif isinstance(arg1, sp.sparse.base.spmatrix):
                # TODO: might want to use more efficient rsb_mtx_alloc_from_csc_const(), rsb_mtx_alloc_from_csr_const()
                (I,J,V)=sp.sparse.find(arg1)
                if dtype is None:
                    self.dtypeA = _dt2dt(arg1.dtype)
            elif isinstance(arg1, tuple):
                if len(arg1) == 2 and not isinstance(arg1[1], tuple):
                    shape=[arg1[0], arg1[1]]
                else:
                    if len(arg1) == 2:
                        # (data, ij) format
                        V = arg1[0]
                        I = arg1[1][0]
                        J = arg1[1][1]
                        if min(J) < 0:
                            raise ValueError('negative J index found')
                        if min(I) < 0:
                            raise ValueError('negative I index found')
                    elif len(arg1) == 3:
                        V = arg1[0]
                        J = arg1[1]
                        P = arg1[2]
                    else:
                        raise ValueError("unrecognized %s_matrix constructor usage"% self.format)
            else:
                try:
                    arg1 = np.asarray(arg1)
                    if dtype is None:
                        self.dtypeA = _dt2dt(arg1.dtype)
                except Exception as e:
                    raise ValueError("unrecognized {}_matrix constructor usage" "".format(self.format)) from e
                (I,J,V)=sp.sparse.find(csr_matrix(arg1))

        if self.dtypeA is None:
            self.dtypeA = rsb_dtype

        if V is None:
            V = []
            I = []
            J = []
        self.typecode = _dt2tc(self.dtype)

        if shape is None:
            shape=[0,0]
            if len(I):
                shape=[max(I)+1,max(J)+1]

        self.nrA = shape[0]
        self.ncA = shape[1]
        self.flagsA = self.flagsA + self._psf2lsf(sym)
        VA = np.array(V,dtype=self.dtype)
        JA = np.array(J,dtype=self.idx_dtype)
        self.nnzA = len(VA)
        assert len(JA) == self.nnzA
        if P is not None:
            IP = np.array(P,dtype=self.idx_dtype)
            assert len(IP) == self.nrA+1 or self.nrA == 0 # recall nrA might be zero (auto)
            self.mtxAp = lr.rsb_mtx_alloc_from_csr_const(<lr.cvoid_ptr> VA.data,<const lr.rsb_coo_idx_t*>IP.data,<const lr.rsb_coo_idx_t*>JA.data,self.nnzA,self.typecode,self.nrA,self.ncA,brA,bcA,self.flagsA,&errval)
        else:
            IP = np.array(I,dtype=self.idx_dtype)
            assert len(IP) == self.nnzA
            self.mtxAp = lr.rsb_mtx_alloc_from_coo_const(<lr.cvoid_ptr> VA.data,<const lr.rsb_coo_idx_t*>IP.data,<const lr.rsb_coo_idx_t*>JA.data,self.nnzA,self.typecode,self.nrA,self.ncA,brA,bcA,self.flagsA,&errval)
        _err_check(errval,want_strict=True)
        self._refresh()
        return
    
    def __str__(self):
        """Return a brief matrix description string."""
        cdef lr.rsb_err_t errval
        cdef size_t buflen = 256
        cdef char buf[256]
        cdef bytes info = b"["
        errval = lr.rsb_mtx_get_info_str(self.mtxAp, "RSB_MIF_MATRIX_INFO__TO__CHAR_P", buf, buflen)
        _err_check(errval)
        # self.do_print()
        info += buf
        info += b"]"
        return str(info)

    def do_print(self, brief=False):
        """
        Print the entire matrix (FIXME: currently, to stdout).
        (specific to rsb).
        """
        if (brief):
            print(self.__str__())
        else:
            return self.save()

    def _mtx_free(self):
        """
        Free the librsb matrix.
        (specific to rsb).
        """
        # print("Freeing matrix.")
        lr.rsb_mtx_free(self.mtxAp)
        self.mtxAp = NULL

    def __dealloc__(self):
        """Destructor."""
        self._mtx_free()

    @property
    def shape(self):
        """
        Shape of the matrix.
        """
        return (self.nrA,self.ncA)

    def __richcmp__(self,f,b):
        """Unfinished."""
	# 0: <
	# 1: <=
	# 2: ==
	# 3: !=
	# 4: >
	# 5: >=
        return False

    cdef _build_from_ptr(self, lr.rsb_mtx_ptr mtxAp):
        """
        Temporary to get entire matrix from mere pointer.
        Might eventually finish in __init__.
        (specific to rsb).
        """
        rm = rsb_matrix(None,dtype=self.dtype)
        rm._mtx_free()
        rm.mtxAp = mtxAp
        rm._refresh()
        return rm

    def _spmul(self, rsb_matrix other):
        """
        Multiply two rsb_matrix objects.
        See scipy.sparse.spmatrix._mul_vector().
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        cdef np.ndarray talpha = np.array([1.0],dtype=self.dtype)
        cdef np.ndarray tbeta = np.array([1.0],dtype=self.dtype)
        cdef lr.rsb_trans_t transA=lr.RSB_TRANSPOSITION_N
        cdef lr.rsb_trans_t transB=lr.RSB_TRANSPOSITION_N
        cdef lr.rsb_flags_t flagsA = lr.RSB_FLAG_NOFLAGS
        mtxBp = lr.rsb_spmsp(self.typecode,transA,talpha.data,self.mtxAp,transB,tbeta.data,other.mtxAp,&errval)
        _err_check(errval)
        return self._build_from_ptr(mtxBp)

    def rescaled(self, alpha):
        """
        Return rescaled copy.
        (specific to rsb).
        """
        cdef rsb_matrix rm = self.copy()
        cdef np.ndarray talpha = np.array([alpha],dtype=self.dtype)
        errval = lr.rsb_mtx_upd_vals(rm.mtxAp,lr.RSB_ELOPF_MUL,talpha.data)
        _err_check(errval)
        return rm

    def rescale(self, alpha):
        """
        Rescale this matrix.
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        cdef np.ndarray talpha = np.array([alpha],dtype=self.dtype)
        errval = lr.rsb_mtx_upd_vals(self.mtxAp,lr.RSB_ELOPF_MUL,talpha.data)
        _err_check(errval)
        return True

    def __mul__(self, x):
        """
           Multiply by a scalar, dense vector, dense matrix (multivector) or another sparse matrix.
           In the case of a scalar, will return a scaled copy of this matrix.
           In the case of a multivector, order is taken from the operand array; C (rows-first) order is recommended with librsb-1.3, otherwise F (columns-first).
           In the case of another sparse matrix, this must be conformant in size.
        """
        cdef np.ndarray y
        if type(x) is type(int(1)):
            return self.__mul__(rsb_dtype(x))
        if type(x) is type(rsb_dtype(1)):
            return self.rescaled(x)
        if type(x) is type(self):
            return self._spmul(x)
        if x.ndim is 1:
            y = np.zeros([self.nr()         ],dtype=self.dtype)
            self._spmv(x,y)
        if x.ndim is 2:
            nrhs=x.shape[1]
            corder = x.flags.c_contiguous
            if corder:
                order='C'
            else:
                order='F'
            y = np.zeros([self.nr(),nrhs],dtype=self.dtype,order=order)
            self._spmm(x,y)
        return y

    def dot(self, x):
        """
        Wrapper to __mul__ (the * operator).
        (specific to rsb, unlike scipy.sparse).
        """
        return self.__mul__(x)

    def _spadd(self, rsb_matrix other):
        """
        Add two rsb_matrix objects.
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        cdef np.ndarray talpha = np.array([1.0],dtype=self.dtype)
        cdef np.ndarray tbeta = np.array([1.0],dtype=self.dtype)
        cdef lr.rsb_trans_t transA=lr.RSB_TRANSPOSITION_N
        cdef lr.rsb_trans_t transB=lr.RSB_TRANSPOSITION_N
        cdef lr.rsb_flags_t flagsA = lr.RSB_FLAG_NOFLAGS
        mtxBp = lr.rsb_sppsp(self.typecode,transA,talpha.data,self.mtxAp,transB,tbeta.data,other.mtxAp,&errval)
        _err_check(errval)
        return self._build_from_ptr(mtxBp)

    def __add__(self,other):
        """Add two rsb_matrix objects (also in scipy.sparse)."""
        return self._spadd(other)

    def opt_set(self, char * opnp, char * opvp):
        """
        Specify individual library options in order to fine-tune the library behaviour.
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        errval = lr.rsb_lib_set_opt_str(opnp,opvp)
        _err_check(errval,want_strict=True)
        return True

    def _otn2obc(self,corder,transA,nrhs):
        """
        Compute operands' leading dimensions.
        """
        cdef lr.rsb_flags_t lr_order = lr.RSB_FLAG_NOFLAGS
        if not corder:
            lr_order=lr.RSB_FLAG_WANT_COLUMN_MAJOR_ORDER
            if self._prt2lt(transA) == lr.RSB_TRANSPOSITION_N:
                ldB=self.ncA
                ldC=self.nrA
            else:
                ldB=self.nrA
                ldC=self.ncA
        else:
            lr_order=lr.RSB_FLAG_WANT_ROW_MAJOR_ORDER
            ldB=nrhs
            ldC=nrhs
        return (lr_order,ldB,ldC)

    def _o2o(self, order):
        cdef lr.rsb_flags_t lr_order = lr.RSB_FLAG_NOFLAGS
        if order in [ b'F', 'F', ord('F') ]:
            lr_order=lr.RSB_FLAG_WANT_COLUMN_MAJOR_ORDER
        else:
            if order in [ b'C', 'C', ord('C') ]:
                lr_order=lr.RSB_FLAG_WANT_ROW_MAJOR_ORDER
            else:
                raise ValueError("Unrecognized order")
        return lr_order

    def autotune(self, lr.rsb_int_t tn=0, lr.rsb_int_t maxr=1, lr.rsb_time_t tmax=2.0, transA='N', alpha=1.0, lr.rsb_coo_idx_t nrhs=1, order='C', beta=1.0, verbose = False):
        """
        Auto-tuner based on rsb_tune_spmm(): optimize either the matrix instance, the thread count or both for rsb_spmm() .
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        cdef lr.rsb_nnz_idx_t ldB=0, ldC=0
        cdef lr.rsb_trans_t transA_ = self._prt2lt(transA)
        cdef lr.rsb_flags_t lr_order = self._o2o(order)
        cdef lr.rsb_real_t sf=1.0
        cdef np.ndarray talpha = np.array([alpha],dtype=self.dtype)
        cdef np.ndarray tbeta = np.array([beta],dtype=self.dtype)
        if (verbose == True):
            self.opt_set(b"RSB_IO_WANT_VERBOSE_TUNING",b"1")
        errval = lr.rsb_tune_spmm(&self.mtxAp,&sf,&tn,maxr,tmax,transA_,talpha.data,NULL,nrhs,lr_order,NULL,ldB,tbeta.data,NULL,ldC);
        assert lr_order==lr.RSB_FLAG_WANT_COLUMN_MAJOR_ORDER or lr_order==lr.RSB_FLAG_WANT_ROW_MAJOR_ORDER
        _err_check(errval)
        self._refresh()
        if (verbose == True):
            self.opt_set(b"RSB_IO_WANT_VERBOSE_TUNING",b"0")
        return sf

    def _find_block(self,frA,lrA,fcA,lcA):
        """
        Extract sparse block as COO.
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        cdef lr.rsb_nnz_idx_t rnz = 0
        cdef lr.rsb_nnz_idx_t*rnzp = &rnz
        errval = lr.rsb_mtx_get_coo_block(self.mtxAp,NULL,NULL,NULL,frA,lrA,fcA,lcA,NULL,NULL,rnzp,lr.RSB_FLAG_NOFLAGS)
        _err_check(errval)
        cdef np.ndarray VAa = np.arange(rnz,dtype=self.dtype)
        cdef np.ndarray JAa = np.arange(rnz,dtype=self.idx_dtype)
        cdef np.ndarray IAa = np.arange(rnz,dtype=self.idx_dtype)
        cdef lr.void_ptr VA = <lr.void_ptr> VAa.data
        cdef lr.rsb_coo_idx_t *IA = <lr.rsb_coo_idx_t*> IAa.data
        cdef lr.rsb_coo_idx_t *JA = <lr.rsb_coo_idx_t*> JAa.data
        errval = lr.rsb_mtx_get_coo_block(self.mtxAp,VA,IA,JA,frA,lrA,fcA,lcA,NULL,NULL,NULL,lr.RSB_FLAG_NOFLAGS)
        _err_check(errval)
        return (np.array(IAa),np.array(JAa),np.array(VAa))

    def getnnz(self):
        return self.nnzA

    @property
    def has_sorted_indices(self):
        """Unfinished."""
        return False

    @property
    def dtype(self):
        return self.dtypeA

    @property
    def nnz(self):
        """
        Number of nonzero entries.
        (specific to rsb).
        """
        return self.getnnz()

    def nsubm(self):
        """
        Number of sparse blocks.
        (specific to rsb).
        """
        return self.nsubmA

    @property
    def _total_size(self):
        """
        (specific to rsb).
        """
        return self.total_size

    def _idx_bpnz(self):
        """
        Index storage bytes per nonzero.
        (specific to rsb).
        """
        return self.idx_bpnz

    def nr(self):
        """
        Number of rows.
        (specific to rsb).
        """
        return self.nrA

    def nc(self):
        """
        Number of columns.
        (specific to rsb).
        """
        return self.ncA

    def _is_complex(self):
        """
        Complex scalar type?
        (specific to rsb).
        """
        return _is_complex_rsb_supported(self.dtype)

    def _is_unsymmetric(self):
        """
        RSB matrix symmetry.
        (specific to rsb).
        """
        if self._get_symchar() == 'G':
            return True
        else:
            return False

    def _refresh(self):
        """Refresh cached variables. (specific to rsb). Candidate for removal."""
        cdef lr.rsb_err_t errval = lr.RSB_ERR_NO_ERROR
        errval |= lr.rsb_mtx_get_info(self.mtxAp, lr.RSB_MIF_MATRIX_ROWS__TO__RSB_COO_INDEX_T,&self.nrA)
        errval |= lr.rsb_mtx_get_info(self.mtxAp, lr.RSB_MIF_MATRIX_COLS__TO__RSB_COO_INDEX_T,&self.ncA)
        errval |= lr.rsb_mtx_get_info(self.mtxAp, lr.RSB_MIF_MATRIX_NNZ__TO__RSB_NNZ_INDEX_T,&self.nnzA)
        errval |= lr.rsb_mtx_get_info(self.mtxAp, lr.RSB_MIF_MATRIX_TYPECODE__TO__RSB_TYPE_T,&self.typecode)
        errval |= lr.rsb_mtx_get_info(self.mtxAp, lr.RSB_MIF_LEAVES_COUNT__TO__RSB_BLK_INDEX_T,&self.nsubmA)
        errval |= lr.rsb_mtx_get_info(self.mtxAp, lr.RSB_MIF_INDEX_STORAGE_IN_BYTES_PER_NNZ__TO__RSB_REAL_T,&self.idx_bpnz)
        errval |= lr.rsb_mtx_get_info(self.mtxAp, lr.RSB_MIF_TOTAL_SIZE__TO__SIZE_T,&self.total_size)
        _err_check(errval,want_strict=True)

    def find(self):
        """
        More or less as scipy.sparse.find(): returns (ia,ja,va).
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        cdef np.ndarray VAa = np.arange(self.nnzA,dtype=self.dtype)
        cdef np.ndarray IAa = np.arange(self.nnzA,dtype=self.idx_dtype)
        cdef np.ndarray JAa = np.arange(self.nnzA,dtype=self.idx_dtype)
        cdef lr.void_ptr VA = <lr.void_ptr> VAa.data
        cdef lr.rsb_coo_idx_t *IA = <lr.rsb_coo_idx_t*> IAa.data
        cdef lr.rsb_coo_idx_t *JA = <lr.rsb_coo_idx_t*> JAa.data
        errval = lr.rsb_mtx_get_coo(self.mtxAp,VA,IA,JA,lr.RSB_FLAG_NOFLAGS)
        _err_check(errval)
        return (np.array(IAa),np.array(JAa),np.array(VAa))

    def _find_v_ij(self):
        (IA,JA,VA)=self.find()
        return (VA,(IA,JA))

    def matvec(self, other):
        """Multiply matrix by vector."""
        return self * other

    def tocsr(self,copy=False):
        """Transition solution (does not exploit rsb_mtx_get_csr)."""
        return csr_matrix(self._find_v_ij())

    def nonzero(self):
        """
        Returns non-zero elements indices.
        Just as scipy.sparse.nonzero().
        """
        (IA,JA,VA)=self.find()
        return (IA,JA)

    def tril(self):
        """
        Just as scipy.sparse.tril().
        """
        [I,J,V]=self.find()
        return sp.sparse.find(sp.sparse.tril(csr_matrix((V,(I,J)))))

    def triu(self):
        """
        Just as scipy.sparse.triu().
        """
        (I,J,V)=self.find()
        return sp.sparse.find(sp.sparse.triu(csr_matrix((V,(I,J)))))

    def issparse(self):
        """Returns True."""
        return True
    def isspmatrix(self):
        """Returns True."""
        return True
    def isspmatrix_csc(self):
        """Returns False."""
        return False
    def isspmatrix_csr(self):
        """Returns False."""
        return False
    def isspmatrix_bsr(self):
        """Returns False."""
        return False
    def isspmatrix_lil(self):
        """Returns False."""
        return False
    def isspmatrix_dok(self):
        """Returns False."""
        return False
    def isspmatrix_coo(self):
        """Returns False."""
        return False
    def isspmatrix_dia(self):
        """Returns False."""
        return False
    def isspmatrix_rsb(self):
        """
        Returns True.
        (specific to rsb).
        """
        return True

    def render(self, filename=None):
        """
        Render block structure to a specified file, in the Encapsulated Postscript (EPS) format.
        With None filename, write to stdout.
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        cdef lr.rsb_coo_idx_t pmWidth=512, pmHeight=512
        cdef lr.rsb_marf_t rflags = lr.RSB_MARF_EPS_B
        if filename is None:
            errval = lr.rsb_mtx_rndr(NULL, self.mtxAp, pmWidth, pmHeight, rflags)
        else:
            if isinstance(filename, bytes):
                pass
            elif isinstance(filename, str):
                filename = bytes(filename, encoding="utf-8")
            else:
                raise TypeError("Unsupported string type")
            errval = lr.rsb_mtx_rndr(filename, self.mtxAp, pmWidth, pmHeight, rflags)
        _err_check(errval)
        return True

    def save(self, char * filename=NULL):
        """
        Save to a specified file, in the Matrix Market format.
        With NULL filename, write to stdout.
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        errval = lr.rsb_file_mtx_save(self.mtxAp,filename)
        _err_check(errval)
        return True

    def copy(self):
        """
        Return a copy of this matrix.
        No data/indices will be shared between the returned value and current matrix.
        (as in scipy.sparse).
        """
        cdef lr.rsb_err_t errval
        cdef lr.rsb_mtx_ptr mtxBp = NULL
        cdef np.ndarray talpha = np.array([1.0],dtype=self.dtype)
        cdef lr.rsb_trans_t transA=lr.RSB_TRANSPOSITION_N
        cdef lr.rsb_flags_t flagsA = lr.RSB_FLAG_NOFLAGS
        errval = lr.rsb_mtx_clone(&mtxBp,self.typecode,transA,talpha.data,self.mtxAp,flagsA)
        _err_check(errval)
        return self._build_from_ptr(mtxBp)

    def todense(self,order=None,out=None):
        """
        Return a dense copy of this matrix.
        (as in scipy.sparse).
        """
        cdef lr.rsb_err_t errval
        cdef lr.rsb_mtx_ptr mtxBp = NULL
        cdef np.ndarray talpha = np.array([1.0],dtype=self.dtype)
        cdef lr.rsb_trans_t transA=lr.RSB_TRANSPOSITION_N
        cdef lr.rsb_flags_t flagsA = lr.RSB_FLAG_NOFLAGS
        #cdef np.ndarray b = np.zeros([self.nrA,self.ncA],dtype=self.dtype)
        cdef np.ndarray b = np.ascontiguousarray(np.zeros([self.nrA,self.ncA],dtype=self.dtype))
        cdef lr.rsb_bool_t rowmajorB
        cdef lr.rsb_nnz_idx_t ldB, nrB, ncB
        if ( order is not 'C' ) and ( order is not 'F' ):
            order='C'
        if order is 'C':
            rowmajorB = lr.RSB_BOOL_TRUE
            ldB=self.ncA; nrB=self.nrA; ncB=self.ncA
        else:
            rowmajorB = lr.RSB_BOOL_FALSE
            ldB=self.nrA; nrB=self.nrA; ncB=self.ncA
        errval = lr.rsb_mtx_add_to_dense(talpha.data,self.mtxAp,ldB,nrB,ncB,rowmajorB,b.data)
        _err_check(errval)
        return b

    def _mini_self_print_test(self):
        """ Candidate for removal."""
        print("*")
        print(self)
        print("*")
        print("a:")
        print(self.find())
        print("a's (1,1):")
        print(self._find_block(1, 1, 1, 1))
        print("a's tril")
        print(self.tril())
        print("a's triu")
        print(self.triu())
        print(" ")

import rsb
rsb.rsb_lib_init()

# vim:et:shiftwidth=4
