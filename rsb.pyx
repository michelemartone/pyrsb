"""
Recursive Sparse Blocks matrix format.
librsb interface for Python.
Proof of concept, very limited interface code, aims at compatibility with scipy.sparse.
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

verbose=0

cpdef rsb_lib_init():
    """Initialize librsb."""
    if verbose:
        print("Initializing librsb")
    return lr.rsb_lib_init(NULL)

cpdef rsb_lib_exit():
    """Finalize librsb."""
    if verbose:
        print("Finalizing librsb")
    return lr.rsb_lib_exit(NULL)

cpdef rsb_file_mtx_load(const char * filename):
    """Load an rsb_matrix matrix from a Matrix Market file."""
    cdef lr.rsb_err_t errval
    rm = rsb_matrix()
    lr.rsb_mtx_free(rm.mtxAp) # workaround: shall maybe pass string to rsb_matrix ?
    rm.mtxAp = lr.rsb_file_mtx_load(filename,rm.flagsA,rm.typecode,&errval)
    _err_check(errval)
    rm._refresh()
    return rm

cpdef rsb_time():
    """Return current time."""
    cdef lr.rsb_time_t rt
    rt = <lr.rsb_time_t>lr.rsb_time()
    return rt

def _print_vec(np.ndarray[np.float_t, ndim=2] x, mylen=0):
    """Print a vector, possibly overriding its length (which is DANGEROUS)."""
    cdef lr.rsb_coo_idx_t ylv = 0
    cdef lr.rsb_type_t typecode = lr.RSB_NUMERICAL_TYPE_DOUBLE
    ylv = len(x)
    if mylen is not 0:
        ylv = mylen
    return lr.rsb_file_vec_save(NULL, typecode, <lr.cvoid_ptr>x.data, ylv)

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
        print "Error reported by librsb: ", str(buf,'ascii')
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
    cdef lr.rsb_flags_t flagsA
    dtype = np.float64
    ndim = 2
    format = 'rsb'

    def _get_dtype(self):
        return self.dtype

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

    def _spmm(self,np.ndarray[np.float_t, ndim=2] x, np.ndarray[np.float_t, ndim=2] y, transA='N', double alpha = 1.0, double beta = 1.0):
        """
        Sparse Matrix by matrix product based on rsb_spmm().
        """
        cdef lr.rsb_coo_idx_t nrhs = x.shape[1]
        cdef lr.rsb_nnz_idx_t ldB, ldC
        cdef lr.rsb_trans_t transA_ = self._prt2lt(transA)
        cdef lr.rsb_flags_t lr_order = lr.RSB_FLAG_NOFLAGS
        cdef lr.rsb_err_t errval
        corder =  x.flags.c_contiguous
        (lr_order,ldB,ldC)=self._otn2obc(corder,transA,nrhs)
        assert x.flags.c_contiguous == y.flags.c_contiguous
        assert lr_order==lr.RSB_FLAG_WANT_COLUMN_MAJOR_ORDER or lr_order==lr.RSB_FLAG_WANT_ROW_MAJOR_ORDER
        if x.shape[1] is not y.shape[1]:
           errval = lr.RSB_ERR_BADARGS
        else:
           errval = lr.rsb_spmm(transA_, &alpha, self.mtxAp, nrhs, lr_order, <lr.cvoid_ptr>x.data, ldB, &beta, <lr.void_ptr>y.data, ldC);
        _err_check(errval)
        return errval

    def _spmv(self,np.ndarray[np.float_t, ndim=1] x, np.ndarray[np.float_t, ndim=1] y, transA='N', double alpha = 1.0, double beta = 1.0):
        """
        Sparse Matrix by vector product based on rsb_spmv().
        """
        cdef lr.rsb_coo_idx_t incX = 1, incY = 1
        cdef lr.rsb_trans_t transA_ = self._prt2lt(transA)
        cdef lr.rsb_err_t errval
        errval = lr.rsb_spmv(transA_, &alpha, self.mtxAp, <lr.cvoid_ptr>x.data, incX, &beta, <lr.void_ptr>y.data, incY)
        _err_check(errval)
        return errval

    def __init__(self,arg1=None,shape=None,sym='U',dtype='d'):
        self.nrA=0
        self.ncA=0
        cdef lr.rsb_blk_idx_t brA = 0, bcA = 0
        cdef lr.cvoid_ptr VA = NULL
        cdef lr.rsb_flags_t flagsA = lr.RSB_FLAG_NOFLAGS
        self.flagsA = flagsA
        self.mtxAp = NULL
        self.typecode = lr.RSB_NUMERICAL_TYPE_DOUBLE
        self.nnzA=0
        V = None
        I = None
        J = None
        if dtype != 'd' and dtype != 'D':
            raise TypeError("Wrong data type: for now, only 'D' suppurted.")
        if arg1 is not None:
            if type(arg1) == type(self):
                self = arg1.copy()
                return
            elif isinstance(arg1, sp.sparse.base.spmatrix):
                # TODO: might want to use more efficient rsb_mtx_alloc_from_csc_const(), rsb_mtx_alloc_from_csr_const()
                (I,J,V)=sp.sparse.find(arg1)
            elif isinstance(arg1, tuple):
                if len(arg1) == 2 and not isinstance(arg1[1], tuple):
                    shape=[arg1[0], arg1[1]]
                else:
                    if len(arg1) == 2:
                        # (data, ij) format
                        V = arg1[0]
                        I = arg1[1][0]
                        J = arg1[1][1]
                    elif len(arg1) == 3:
                        # TODO: might want to use more efficient rsb_mtx_alloc_from_csr_const()
                        # (data, indices, indptr) format
                        # raise ValueError("unrecognized %s_matrix constructor usage"% self.format)
                        # here: (data, i,j) format
                        V = arg1[0]
                        J = arg1[1]
                        P = arg1[2]
                        [I,J,V]=sp.sparse.find(csr_matrix((V,J,P)))
                    else:
                        raise ValueError("unrecognized %s_matrix constructor usage"% self.format)
            else:
                try:
                    arg1 = np.asarray(arg1)
                except Exception as e:
                    raise ValueError("unrecognized {}_matrix constructor usage" "".format(self.format)) from e
                (I,J,V)=sp.sparse.find(csr_matrix(arg1))

        if V is None:
            V = []
            I = []
            J = []

        if shape is None:
            shape=[0,0]
            if len(I):
                shape=[max(I)+1,max(J)+1]

        self.nrA=shape[0]
        self.ncA=shape[1]
        self.flagsA = self.flagsA + self._psf2lsf(sym)
        cdef lr.rsb_err_t errval
        cdef lr.rsb_coo_idx_t*IA = NULL, *JA = NULL
        cdef np.ndarray VAa = np.array(V,dtype=np.double)
        cdef np.ndarray IAa = np.array(I,dtype=np.int32)
        cdef np.ndarray JAa = np.array(J,dtype=np.int32)
        self.nnzA=len(VAa)
        VA=<lr.void_ptr> VAa.data
        IA=<lr.rsb_coo_idx_t*>IAa.data
        JA=<lr.rsb_coo_idx_t*>JAa.data
        self.mtxAp = lr.rsb_mtx_alloc_from_coo_const(VA,IA,JA,self.nnzA,self.typecode,self.nrA,self.ncA,brA,bcA,self.flagsA,&errval)
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
        Free the matrix.
        """
        # print("Freeing matrix.")
        lr.rsb_mtx_free(self.mtxAp)
        self.mtxAp = NULL

    def __dealloc__(self):
        """Destructor."""
        self._mtx_free()

    def getshape(self):
        """
        Shape of the matrix.
        """
        return (self.nrA,self.ncA)

    @property
    def shape(self):
        """
        Shape of the matrix.
        """
        return self.getshape()

    def __richcmp__(self,f,b):
        """Unfinished."""
	# 0: <
	# 1: <=
	# 2: ==
	# 3: !=
	# 4: >
	# 5: >=
        return False

    def _spmul(self, rsb_matrix other):
        """
        Multiply two rsb_matrix objects.
        (specific to rsb; __mul__ with scipy).
        """
        cdef lr.rsb_err_t errval
        cdef double alpha = 1.0, beta = 1.0
        cdef lr.rsb_trans_t transA=lr.RSB_TRANSPOSITION_N
        cdef lr.rsb_trans_t transB=lr.RSB_TRANSPOSITION_N
        cdef lr.rsb_flags_t flagsA = lr.RSB_FLAG_NOFLAGS
        rm = rsb_matrix()
        rm.mtxAp = lr.rsb_spmsp(self.typecode,transA,&alpha,self.mtxAp,transB,&beta,other.mtxAp,&errval)
        _err_check(errval)
        rm._refresh()
        return rm

    def rescaled(self, double alpha):
        """
        Return rescaled copy.
        (specific to rsb).
        """
        rm = self.copy()
        rm.rescale(alpha)
        return rm

    def rescale(self, double alpha):
        """
        Rescale this matrix.
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        errval = lr.rsb_mtx_upd_vals(self.mtxAp,lr.RSB_ELOPF_MUL,&alpha)
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
            return self.__mul__(float(x))
        if type(x) is type(float(1)):
            return self.rescaled(x)
        if type(x) is type(self):
            return self._spmul(x)
        if x.ndim is 1:
            y = np.zeros([self.nr()         ],dtype=np.double)
            self._spmv(x,y)
        if x.ndim is 2:
            nrhs=x.shape[1]
            corder = x.flags.c_contiguous
            if corder:
                order='C'
            else:
                order='F'
            y = np.zeros([self.nr(),nrhs],dtype=np.double,order=order)
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
        """
        cdef lr.rsb_err_t errval
        cdef double alpha = 1.0, beta = 1.0
        cdef lr.rsb_trans_t transA=lr.RSB_TRANSPOSITION_N
        cdef lr.rsb_trans_t transB=lr.RSB_TRANSPOSITION_N
        cdef lr.rsb_flags_t flagsA = lr.RSB_FLAG_NOFLAGS
        rm = rsb_matrix()
        rm.mtxAp = lr.rsb_sppsp(self.typecode,transA,&alpha,self.mtxAp,transB,&beta,other.mtxAp,&errval)
        _err_check(errval)
        rm._refresh()
        return rm

    def __add__(self,other):
        """Add two rsb_matrix objects (also in scipy.sparse)."""
        return self._spadd(other)

    def __complex__(self,other):
        """Unsupported: at the moment only double is supported."""
        return False

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
        cdef lr.rsb_flags_t lr_order = lr.RSB_FLAG_NOFLAGS
        if not corder:
            lr_order=lr.RSB_FLAG_WANT_COLUMN_MAJOR_ORDER
            if transA == b'N':
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

    def _o2o(self, lr.rsb_flags_t order):
        cdef lr.rsb_flags_t lr_order = lr.RSB_FLAG_NOFLAGS
        if order == b'F':
            lr_order=lr.RSB_FLAG_WANT_COLUMN_MAJOR_ORDER
        else:
            if order == b'C':
                lr_order=lr.RSB_FLAG_WANT_ROW_MAJOR_ORDER
            else:
                assert False
        return lr_order

    def autotune(self, lr.rsb_real_t sf=1.0, lr.rsb_int_t tn=0, lr.rsb_int_t maxr=1, lr.rsb_time_t tmax=2.0, lr.rsb_trans_t transA=b'N', double alpha=1.0, lr.rsb_coo_idx_t nrhs=1, lr.rsb_flags_t order=b'F', double beta=1.0, verbose = False):
        """
        Auto-tuner based on rsb_tune_spmm(): optimize either the matrix instance, the thread count or both for rsb_spmm() .
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        cdef lr.rsb_nnz_idx_t ldB=0, ldC=0
        cdef lr.rsb_trans_t transA_ = self._prt2lt(transA)
        cdef lr.rsb_flags_t lr_order = self._o2o(order)
        if (verbose == True):
            self.opt_set(b"RSB_IO_WANT_VERBOSE_TUNING",b"1")
        errval = lr.rsb_tune_spmm(&self.mtxAp,&sf,&tn,maxr,tmax,transA_,&alpha,NULL,nrhs,lr_order,NULL,ldB,&beta,NULL,ldC);
        assert lr_order==lr.RSB_FLAG_WANT_COLUMN_MAJOR_ORDER or lr_order==lr.RSB_FLAG_WANT_ROW_MAJOR_ORDER
        _err_check(errval)
        if (verbose == True):
            self.opt_set(b"RSB_IO_WANT_VERBOSE_TUNING",b"0")
        return True

    def _find_block(self,frA,lrA,fcA,lcA):
        """
        Extract sparse block as COO.
        Unfinished.
        """
        cdef lr.rsb_err_t errval
        cdef lr.rsb_nnz_idx_t rnz = 0
        cdef lr.rsb_nnz_idx_t*rnzp = &rnz
        errval = lr.rsb_mtx_get_coo_block(self.mtxAp,NULL,NULL,NULL,frA,lrA,fcA,lcA,NULL,NULL,rnzp,lr.RSB_FLAG_NOFLAGS)
        _err_check(errval)
        cdef np.ndarray VAa = np.arange(rnz,dtype=np.double)
        cdef np.ndarray JAa = np.arange(rnz,dtype=np.int32)
        cdef np.ndarray IAa = np.arange(rnz,dtype=np.int32)
        cdef lr.void_ptr VA = NULL
        cdef lr.rsb_coo_idx_t*IA = NULL, *JA = NULL
        VA=<lr.void_ptr> VAa.data
        IA=<lr.rsb_coo_idx_t*> IAa.data
        JA=<lr.rsb_coo_idx_t*> JAa.data
        errval = lr.rsb_mtx_get_coo_block(self.mtxAp,VA,IA,JA,frA,lrA,fcA,lcA,NULL,NULL,NULL,lr.RSB_FLAG_NOFLAGS)
        _err_check(errval)
        return (np.array(IAa),np.array(JAa),np.array(VAa))

    def getnnz(self):
        return self.nnzA

    @property
    def has_sorted_indices(self):
        return False

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

    def _is_unsymmetric(self):
        """
        RSB matrix symmetry.
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        cdef lr.rsb_flags_t flagsA = lr.RSB_FLAG_NOFLAGS
        errval = lr.rsb_mtx_get_info(self.mtxAp, lr.RSB_MIF_MATRIX_FLAGS__TO__RSB_FLAGS_T,&flagsA)
        if ( ( flagsA & (lr.RSB_FLAG_HERMITIAN | lr.RSB_FLAG_SYMMETRIC ) ) == lr.RSB_FLAG_NOFLAGS ):
            return True
        else:
            return False

    def _refresh(self):
        cdef lr.rsb_err_t errval
        errval |= lr.rsb_mtx_get_info(self.mtxAp, lr.RSB_MIF_MATRIX_ROWS__TO__RSB_COO_INDEX_T,&self.nrA)
        errval |= lr.rsb_mtx_get_info(self.mtxAp, lr.RSB_MIF_MATRIX_COLS__TO__RSB_COO_INDEX_T,&self.ncA)
        errval |= lr.rsb_mtx_get_info(self.mtxAp, lr.RSB_MIF_MATRIX_NNZ__TO__RSB_NNZ_INDEX_T,&self.nnzA)
        errval |= lr.rsb_mtx_get_info(self.mtxAp, lr.RSB_MIF_MATRIX_TYPECODE__TO__RSB_TYPE_T,&self.typecode)
        errval |= lr.rsb_mtx_get_info(self.mtxAp, lr.RSB_MIF_LEAVES_COUNT__TO__RSB_BLK_INDEX_T,&self.nsubmA)
        _err_check(errval,want_strict=True)

    def find(self):
        """
        More or less as scipy.sparse.find(): returns (ia,ja,va).
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        cdef lr.void_ptr VA = NULL
        cdef lr.rsb_coo_idx_t*IA = NULL, *JA = NULL
        cdef np.ndarray VAa = np.arange(self.nnzA,dtype=np.double)
        cdef np.ndarray IAa = np.arange(self.nnzA,dtype=np.int32)
        cdef np.ndarray JAa = np.arange(self.nnzA,dtype=np.int32)
        VA=<lr.void_ptr> VAa.data
        IA=<lr.rsb_coo_idx_t*> IAa.data
        JA=<lr.rsb_coo_idx_t*> JAa.data
        errval = lr.rsb_mtx_get_coo(self.mtxAp,VA,IA,JA,lr.RSB_FLAG_NOFLAGS)
        _err_check(errval)
        return (np.array(IAa),np.array(JAa),np.array(VAa))

    def _find_v_ij(self):
        (IA,JA,VA)=self.find()
        return (VA,(IA,JA))

    def tocsr(self,copy=False):
        """Transition solution (does not exploit rsb_mtx_get_csr)."""
        return csr_matrix(self._find_v_ij())

    def nonzero(self):
        """
        More or less as csr_matrix.nonzero(): returns (ia,ja).
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

    def save(self, char * filename=NULL):
        """
        Save to a specified file, in the Matrix Market format.
        With NULL Input, to stdout.
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        errval = lr.rsb_file_mtx_save(self.mtxAp,filename)
        _err_check(errval)
        return True

    def copy(self):
        """
        Return a copy (clone) of this matrix.
        (specific to rsb).
        """
        cdef lr.rsb_err_t errval
        cdef lr.rsb_mtx_ptr mtxBp = NULL
        cdef double alpha = 1.0
        cdef lr.rsb_trans_t transA=lr.RSB_TRANSPOSITION_N
        cdef lr.rsb_flags_t flagsA = lr.RSB_FLAG_NOFLAGS
        errval = lr.rsb_mtx_clone(&mtxBp,self.typecode,transA,&alpha,self.mtxAp,flagsA)
        rm = rsb_matrix()
        _err_check(errval)
        rm.mtxAp = mtxBp
        rm._refresh()
        return rm

    def todense(self,order=None,out=None):
        """
        Return a dense copy of this matrix.
        (as in scipy.sparse).
        """
        cdef lr.rsb_err_t errval
        cdef lr.rsb_mtx_ptr mtxBp = NULL
        cdef double alpha = 1.0
        cdef lr.rsb_trans_t transA=lr.RSB_TRANSPOSITION_N
        cdef lr.rsb_flags_t flagsA = lr.RSB_FLAG_NOFLAGS
        #cdef np.ndarray b = np.zeros([self.nrA,self.ncA],dtype=np.double)
        cdef np.ndarray b = np.ascontiguousarray(np.zeros([self.nrA,self.ncA],dtype=np.double))
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
        errval = lr.rsb_mtx_add_to_dense(&alpha,self.mtxAp,ldB,nrB,ncB,rowmajorB,b.data)
        _err_check(errval)
        return b

    def mini_self_print_test(self):
        """Unfinished."""
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
