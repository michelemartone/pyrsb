"""
librsb for python
Proof of concept, very limited interface code.
Author: Michele Martone
"""
cdef extern from "rsb.h":
	ctypedef char* char_ptr "char*"
	ctypedef char* const_char_ptr "const char*"
	ctypedef void* void_ptr  "void*"
	ctypedef void* cvoid_ptr  "const void*"
	ctypedef void* rsb_mtx_ptr  "struct rsb_mtx_t*"
	ctypedef void** rsb_mtx_pptr  "struct rsb_mtx_t**"
	ctypedef void* rsb_mtx_cptr  "const struct rsb_mtx_t*"
	ctypedef void* rsb_opt_ptr  "struct rsb_initopts*"
	
	ctypedef int rsb_opt_t "rsb_opt_t"
	ctypedef int rsb_mif_t "rsb_mif_t"
	ctypedef int rsb_elopf_t "rsb_elopf_t"
	ctypedef int rsb_extff_t "rsb_extff_t"
	ctypedef signed int rsb_blk_idx_t "rsb_blk_idx_t"
	ctypedef signed int rsb_coo_idx_t "rsb_coo_idx_t"
	ctypedef signed int rsb_nnz_idx_t "rsb_nnz_idx_t"
	ctypedef signed int rsb_flags_t "rsb_flags_t"
	ctypedef char rsb_type_t "rsb_type_t"
	ctypedef signed int rsb_err_t "rsb_err_t"
	ctypedef signed int rsb_int_t "rsb_int_t"
	ctypedef rsb_flags_t rsb_bool_t "rsb_bool_t"
	ctypedef rsb_flags_t rsb_trans_t "rsb_trans_t"
	ctypedef double rsb_real_t "rsb_real_t"
	ctypedef char rsb_char_t "rsb_char_t"
	ctypedef rsb_real_t rsb_time_t "rsb_time_t"
	ctypedef rsb_flags_t rsb_marf_t "rsb_marf_t"
	ctypedef rsb_flags_t rsb_precf_t "rsb_precf_t"
	cdef int RSB_BOOL_TRUE "RSB_BOOL_TRUE"
	cdef int RSB_BOOL_FALSE "RSB_BOOL_FALSE"
	cdef int RSB_DEFAULT_ROW_BLOCKING "RSB_DEFAULT_ROW_BLOCKING"
	cdef int RSB_DEFAULT_COL_BLOCKING "RSB_DEFAULT_COL_BLOCKING"
	cdef int RSB_DEFAULT_BLOCKING "RSB_DEFAULT_BLOCKING"
	cdef int RSB_CHAR_BIT "RSB_CHAR_BIT"
	cdef int RSB_MIN_MATRIX_DIM "RSB_MIN_MATRIX_DIM"
	cdef int RSB_MIN_MATRIX_NNZ "RSB_MIN_MATRIX_NNZ"
	cdef int RSB_NNZ_BLK_MAX "RSB_NNZ_BLK_MAX"
	cdef int RSB_MAX_MATRIX_DIM "RSB_MAX_MATRIX_DIM"
	cdef int RSB_MAX_MATRIX_NNZ "RSB_MAX_MATRIX_NNZ"
	cdef int RSB_MARKER_COO_VALUE "RSB_MARKER_COO_VALUE"
	cdef int RSB_MARKER_NNZ_VALUE "RSB_MARKER_NNZ_VALUE"
	cdef int RSB_INVALID_COO_IDX_VAL "RSB_INVALID_COO_IDX_VAL"
	cdef int RSB_INVALID_NNZ_IDX_VAL "RSB_INVALID_NNZ_IDX_VAL"
	cdef int RSB_FLAG_DEFAULT_STORAGE_FLAGS "RSB_FLAG_DEFAULT_STORAGE_FLAGS"
	cdef int RSB_FLAG_DEFAULT_COO_MATRIX_FLAGS "RSB_FLAG_DEFAULT_COO_MATRIX_FLAGS"
	cdef int RSB_FLAG_DEFAULT_CSR_MATRIX_FLAGS "RSB_FLAG_DEFAULT_CSR_MATRIX_FLAGS"
	cdef int RSB_FLAG_DEFAULT_RSB_MATRIX_FLAGS "RSB_FLAG_DEFAULT_RSB_MATRIX_FLAGS"
	cdef int RSB_FLAG_DEFAULT_MATRIX_FLAGS "RSB_FLAG_DEFAULT_MATRIX_FLAGS"
	cdef int RSB_FLAG_NOFLAGS "RSB_FLAG_NOFLAGS"
	cdef int RSB_FLAG_IDENTICAL_FLAGS "RSB_FLAG_IDENTICAL_FLAGS"
	cdef int RSB_FLAG_FORTRAN_INDICES_INTERFACE "RSB_FLAG_FORTRAN_INDICES_INTERFACE"
	cdef int RSB_FLAG_C_INDICES_INTERFACE "RSB_FLAG_C_INDICES_INTERFACE"
	cdef int RSB_FLAG_USE_HALFWORD_INDICES "RSB_FLAG_USE_HALFWORD_INDICES"
	cdef int RSB_FLAG_WANT_ROW_MAJOR_ORDER "RSB_FLAG_WANT_ROW_MAJOR_ORDER"
	cdef int RSB_FLAG_WANT_COLUMN_MAJOR_ORDER "RSB_FLAG_WANT_COLUMN_MAJOR_ORDER"
	cdef int RSB_FLAG_SORTED_INPUT "RSB_FLAG_SORTED_INPUT"
	cdef int RSB_FLAG_TRIANGULAR "RSB_FLAG_TRIANGULAR"
	cdef int RSB_FLAG_LOWER "RSB_FLAG_LOWER"
	cdef int RSB_FLAG_UPPER "RSB_FLAG_UPPER"
	cdef int RSB_FLAG_UNIT_DIAG_IMPLICIT "RSB_FLAG_UNIT_DIAG_IMPLICIT"
	cdef int RSB_FLAG_WANT_COO_STORAGE "RSB_FLAG_WANT_COO_STORAGE"
	cdef int RSB_FLAG_DUPLICATES_KEEP_LAST "RSB_FLAG_DUPLICATES_KEEP_LAST"
	cdef int RSB_FLAG_DUPLICATES_DEFAULT_HANDLE "RSB_FLAG_DUPLICATES_DEFAULT_HANDLE"
	cdef int RSB_FLAG_DUPLICATES_SUM "RSB_FLAG_DUPLICATES_SUM"
	cdef int RSB_FLAG_DISCARD_ZEROS "RSB_FLAG_DISCARD_ZEROS"
	cdef int RSB_FLAG_QUAD_PARTITIONING "RSB_FLAG_QUAD_PARTITIONING"
	cdef int RSB_FLAG_WANT_BCSS_STORAGE "RSB_FLAG_WANT_BCSS_STORAGE"
	cdef int RSB_FLAG_ASSEMBLED_IN_COO_ARRAYS "RSB_FLAG_ASSEMBLED_IN_COO_ARRAYS"
	cdef int RSB_FLAG_EXPERIMENTAL_IN_PLACE_PERMUTATION_SORT "RSB_FLAG_EXPERIMENTAL_IN_PLACE_PERMUTATION_SORT"
	cdef int RSB_FLAG_SYMMETRIC "RSB_FLAG_SYMMETRIC"
	cdef int RSB_FLAG_HERMITIAN "RSB_FLAG_HERMITIAN"
	cdef int RSB_FLAG_RECURSIVE_MORE_LEAVES_THAN_THREADS "RSB_FLAG_RECURSIVE_MORE_LEAVES_THAN_THREADS"
	cdef int RSB_FLAG_LOWER_HERMITIAN "RSB_FLAG_LOWER_HERMITIAN"
	cdef int RSB_FLAG_UPPER_HERMITIAN "RSB_FLAG_UPPER_HERMITIAN"
	cdef int RSB_FLAG_LOWER_TRIANGULAR "RSB_FLAG_LOWER_TRIANGULAR"
	cdef int RSB_FLAG_UPPER_TRIANGULAR "RSB_FLAG_UPPER_TRIANGULAR"
	cdef int RSB_FLAG_LOWER_SYMMETRIC "RSB_FLAG_LOWER_SYMMETRIC"
	cdef int RSB_FLAG_DIAGONAL "RSB_FLAG_DIAGONAL"
	cdef int RSB_FLAG_UPPER_SYMMETRIC "RSB_FLAG_UPPER_SYMMETRIC"
	cdef int RSB_FLAG_RECURSIVE_SUBDIVIDE_MORE_ON_DIAG "RSB_FLAG_RECURSIVE_SUBDIVIDE_MORE_ON_DIAG"
	cdef int RSB_FLAG_EXTERNALLY_ALLOCATED_ARRAYS "RSB_FLAG_EXTERNALLY_ALLOCATED_ARRAYS"
	cdef int RSB_FLAG_USE_CSR_RESERVED "RSB_FLAG_USE_CSR_RESERVED"
	cdef int RSB_FLAG_USE_HALFWORD_INDICES_CSR "RSB_FLAG_USE_HALFWORD_INDICES_CSR"
	cdef int RSB_FLAG_USE_HALFWORD_INDICES_COO "RSB_FLAG_USE_HALFWORD_INDICES_COO"
	cdef int RSB_FLAG_MUTUALLY_EXCLUSIVE_SWITCHES "RSB_FLAG_MUTUALLY_EXCLUSIVE_SWITCHES"
	cdef int RSB_ERR_NO_ERROR "RSB_ERR_NO_ERROR"
	cdef int RSB_ERR_GENERIC_ERROR "RSB_ERR_GENERIC_ERROR"
	cdef int RSB_ERR_UNSUPPORTED_OPERATION "RSB_ERR_UNSUPPORTED_OPERATION"
	cdef int RSB_ERR_UNSUPPORTED_TYPE "RSB_ERR_UNSUPPORTED_TYPE"
	cdef int RSB_ERR_UNSUPPORTED_FORMAT "RSB_ERR_UNSUPPORTED_FORMAT"
	cdef int RSB_ERR_INTERNAL_ERROR "RSB_ERR_INTERNAL_ERROR"
	cdef int RSB_ERR_BADARGS "RSB_ERR_BADARGS"
	cdef int RSB_ERR_ENOMEM "RSB_ERR_ENOMEM"
	cdef int RSB_ERR_UNIMPLEMENTED_YET "RSB_ERR_UNIMPLEMENTED_YET"
	cdef int RSB_ERR_LIMITS "RSB_ERR_LIMITS"
	cdef int RSB_ERR_FORTRAN_ERROR "RSB_ERR_FORTRAN_ERROR"
	cdef int RSB_ERR_UNSUPPORTED_FEATURE "RSB_ERR_UNSUPPORTED_FEATURE"
	cdef int RSB_ERR_NO_USER_CONFIGURATION "RSB_ERR_NO_USER_CONFIGURATION"
	cdef int RSB_ERR_CORRUPT_INPUT_DATA "RSB_ERR_CORRUPT_INPUT_DATA"
	cdef int RSB_ERR_FAILED_MEMHIER_DETECTION "RSB_ERR_FAILED_MEMHIER_DETECTION"
	cdef int RSB_ERR_COULD_NOT_HONOUR_EXTERNALLY_ALLOCATION_FLAGS "RSB_ERR_COULD_NOT_HONOUR_EXTERNALLY_ALLOCATION_FLAGS"
	cdef int RSB_ERR_NO_STREAM_OUTPUT_CONFIGURED_OUT "RSB_ERR_NO_STREAM_OUTPUT_CONFIGURED_OUT"
	cdef int RSB_ERR_INVALID_NUMERICAL_DATA "RSB_ERR_INVALID_NUMERICAL_DATA"
	cdef int RSB_ERR_MEMORY_LEAK "RSB_ERR_MEMORY_LEAK"
	cdef int RSB_ERR_ELEMENT_NOT_FOUND "RSB_ERR_ELEMENT_NOT_FOUND"
	cdef int RSB_ERRS_UNSUPPORTED_FEATURES "RSB_ERRS_UNSUPPORTED_FEATURES"
	cdef int RSB_PROGRAM_SUCCESS "RSB_PROGRAM_SUCCESS"
	cdef int RSB_PROGRAM_ERROR "RSB_PROGRAM_ERROR"
	cdef int RSB_IO_SPECIFIER_GET "RSB_IO_SPECIFIER_GET"
	cdef int RSB_IO_SPECIFIER_SET "RSB_IO_SPECIFIER_SET"
	cdef int RSB_NULL_INIT_OPTIONS "RSB_NULL_INIT_OPTIONS"
	cdef int RSB_NULL_EXIT_OPTIONS "RSB_NULL_EXIT_OPTIONS"
	cdef int RSB_MARF_RGB "RSB_MARF_RGB"
	cdef int RSB_MARF_EPS_S "RSB_MARF_EPS_S"
	cdef int RSB_MARF_EPS_B "RSB_MARF_EPS_B"
	cdef int RSB_MARF_EPS "RSB_MARF_EPS"
	cdef int RSB_MARF_EPS_L "RSB_MARF_EPS_L"
	cdef int RSB_PRECF_ILU0 "RSB_PRECF_ILU0"
	cdef int RSB_IO_WANT_VERBOSE_INIT "RSB_IO_WANT_VERBOSE_INIT"
	cdef int RSB_IO_WANT_VERBOSE_EXIT "RSB_IO_WANT_VERBOSE_EXIT"
	cdef int RSB_IO_WANT_OUTPUT_STREAM "RSB_IO_WANT_OUTPUT_STREAM"
	cdef int RSB_IO_WANT_SORT_METHOD "RSB_IO_WANT_SORT_METHOD"
	cdef int RSB_IO_WANT_CACHE_BLOCKING_METHOD "RSB_IO_WANT_CACHE_BLOCKING_METHOD"
	cdef int RSB_IO_WANT_SUBDIVISION_MULTIPLIER "RSB_IO_WANT_SUBDIVISION_MULTIPLIER"
	cdef int RSB_IO_WANT_VERBOSE_ERRORS "RSB_IO_WANT_VERBOSE_ERRORS"
	cdef int RSB_IO_WANT_BOUNDED_BOX_COMPUTATION "RSB_IO_WANT_BOUNDED_BOX_COMPUTATION"
	cdef int RSB_IO_WANT_EXECUTING_THREADS "RSB_IO_WANT_EXECUTING_THREADS"
	cdef int RSB_IO_WANT_EXTRA_VERBOSE_INTERFACE "RSB_IO_WANT_EXTRA_VERBOSE_INTERFACE"
	cdef int RSB_IO_WANT_MEMORY_HIERARCHY_INFO_STRING "RSB_IO_WANT_MEMORY_HIERARCHY_INFO_STRING"
	cdef int RSB_IO_WANT_IS_INITIALIZED_MARKER "RSB_IO_WANT_IS_INITIALIZED_MARKER"
	cdef int RSB_IO_WANT_MEM_ALLOC_CNT "RSB_IO_WANT_MEM_ALLOC_CNT"
	cdef int RSB_IO_WANT_MEM_ALLOC_TOT "RSB_IO_WANT_MEM_ALLOC_TOT"
	cdef int RSB_IO_WANT_LEAF_LEVEL_MULTIVEC "RSB_IO_WANT_LEAF_LEVEL_MULTIVEC"
	cdef int RSB_IO_WANT_MAX_MEMORY_ALLOCATIONS "RSB_IO_WANT_MAX_MEMORY_ALLOCATIONS"
	cdef int RSB_IO_WANT_MAX_MEMORY_ALLOCATED "RSB_IO_WANT_MAX_MEMORY_ALLOCATED"
	cdef int RSB_IO_WANT_LIBRSB_ETIME "RSB_IO_WANT_LIBRSB_ETIME"
	cdef int RSB_IO_WANT_VERBOSE_TUNING "RSB_IO_WANT_VERBOSE_TUNING"
	cdef int RSB_EXTF_NORM_ONE "RSB_EXTF_NORM_ONE"
	cdef int RSB_EXTF_NORM_TWO "RSB_EXTF_NORM_TWO"
	cdef int RSB_EXTF_NORM_INF "RSB_EXTF_NORM_INF"
	cdef int RSB_EXTF_SUMS_ROW "RSB_EXTF_SUMS_ROW"
	cdef int RSB_EXTF_SUMS_COL "RSB_EXTF_SUMS_COL"
	cdef int RSB_EXTF_ASUMS_ROW "RSB_EXTF_ASUMS_ROW"
	cdef int RSB_EXTF_ASUMS_COL "RSB_EXTF_ASUMS_COL"
	cdef int RSB_EXTF_DIAG "RSB_EXTF_DIAG"
	cdef int RSB_MIF_INDEX_STORAGE_IN_BYTES__TO__SIZE_T "RSB_MIF_INDEX_STORAGE_IN_BYTES__TO__SIZE_T"
	cdef int RSB_MIF_INDEX_STORAGE_IN_BYTES_PER_NNZ__TO__RSB_REAL_T "RSB_MIF_INDEX_STORAGE_IN_BYTES_PER_NNZ__TO__RSB_REAL_T"
	cdef int RSB_MIF_MATRIX_ROWS__TO__RSB_COO_INDEX_T "RSB_MIF_MATRIX_ROWS__TO__RSB_COO_INDEX_T"
	cdef int RSB_MIF_MATRIX_COLS__TO__RSB_COO_INDEX_T "RSB_MIF_MATRIX_COLS__TO__RSB_COO_INDEX_T"
	cdef int RSB_MIF_MATRIX_NNZ__TO__RSB_NNZ_INDEX_T "RSB_MIF_MATRIX_NNZ__TO__RSB_NNZ_INDEX_T"
	cdef int RSB_MIF_TOTAL_SIZE__TO__SIZE_T "RSB_MIF_TOTAL_SIZE__TO__SIZE_T"
	cdef int RSB_MIF_MATRIX_FLAGS__TO__RSB_FLAGS_T "RSB_MIF_MATRIX_FLAGS__TO__RSB_FLAGS_T"
	cdef int RSB_MIF_MATRIX_TYPECODE__TO__RSB_TYPE_T "RSB_MIF_MATRIX_TYPECODE__TO__RSB_TYPE_T"
	cdef int RSB_MIF_MATRIX_INFO__TO__CHAR_P "RSB_MIF_MATRIX_INFO__TO__CHAR_P"
	cdef int RSB_MIF_LEAVES_COUNT__TO__RSB_BLK_INDEX_T "RSB_MIF_LEAVES_COUNT__TO__RSB_BLK_INDEX_T"
	cdef int RSB_ELOPF_MUL "RSB_ELOPF_MUL"
	cdef int RSB_ELOPF_DIV "RSB_ELOPF_DIV"
	cdef int RSB_ELOPF_POW "RSB_ELOPF_POW"
	cdef int RSB_ELOPF_NEG "RSB_ELOPF_NEG"
	cdef int RSB_ELOPF_SCALE_ROWS "RSB_ELOPF_SCALE_ROWS"
	cdef int RSB_ELOPF_SCALE_COLS "RSB_ELOPF_SCALE_COLS"
	cdef int RSB_ELOPF_SCALE_ROWS_REAL "RSB_ELOPF_SCALE_ROWS_REAL"
	cdef int RSB_ELOPF_SCALE_COLS_REAL "RSB_ELOPF_SCALE_COLS_REAL"
	cdef int RSB_COORDINATE_TYPE_C "RSB_COORDINATE_TYPE_C"
	cdef int RSB_COORDINATE_TYPE_H "RSB_COORDINATE_TYPE_H"
	cdef int RSB_TRANSPOSITION_N "RSB_TRANSPOSITION_N"
	cdef int RSB_TRANSPOSITION_T "RSB_TRANSPOSITION_T"
	cdef int RSB_TRANSPOSITION_C "RSB_TRANSPOSITION_C"
	cdef int RSB_TRANSPOSITION_INVALID "RSB_TRANSPOSITION_INVALID"
	cdef int RSB_SYMMETRY_U "RSB_SYMMETRY_U"
	cdef int RSB_SYMMETRY_S "RSB_SYMMETRY_S"
	cdef int RSB_SYMMETRY_H "RSB_SYMMETRY_H"
	cdef int RSB_DIAGONAL_E "RSB_DIAGONAL_E"
	cdef int RSB_DIAGONAL_I "RSB_DIAGONAL_I"
	cdef int RSB_MATRIX_STORAGE_BCOR "RSB_MATRIX_STORAGE_BCOR"
	cdef int RSB_MATRIX_STORAGE_BCSR "RSB_MATRIX_STORAGE_BCSR"
	cdef int RSB_MATRIX_STORAGE_BCOR_STRING "RSB_MATRIX_STORAGE_BCOR_STRING"
	cdef int RSB_MATRIX_STORAGE_BCSR_STRING "RSB_MATRIX_STORAGE_BCSR_STRING"
	cdef int RSB_NUMERICAL_TYPE_DOUBLE "RSB_NUMERICAL_TYPE_DOUBLE"
	cdef int RSB_NUMERICAL_TYPE_FLOAT "RSB_NUMERICAL_TYPE_FLOAT"
	cdef int RSB_NUMERICAL_TYPE_FLOAT_COMPLEX "RSB_NUMERICAL_TYPE_FLOAT_COMPLEX"
	cdef int RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX "RSB_NUMERICAL_TYPE_DOUBLE_COMPLEX"
	cdef int RSB_NUMERICAL_TYPE_FORTRAN_SAME_TYPE "RSB_NUMERICAL_TYPE_FORTRAN_SAME_TYPE"
	cdef int RSB_NUMERICAL_TYPE_FORTRAN_INT "RSB_NUMERICAL_TYPE_FORTRAN_INT"
	cdef int RSB_NUMERICAL_TYPE_FORTRAN_DOUBLE "RSB_NUMERICAL_TYPE_FORTRAN_DOUBLE"
	cdef int RSB_NUMERICAL_TYPE_FORTRAN_FLOAT "RSB_NUMERICAL_TYPE_FORTRAN_FLOAT"
	cdef int RSB_NUMERICAL_TYPE_FORTRAN_FLOAT_COMPLEX "RSB_NUMERICAL_TYPE_FORTRAN_FLOAT_COMPLEX"
	cdef int RSB_NUMERICAL_TYPE_FORTRAN_DOUBLE_COMPLEX "RSB_NUMERICAL_TYPE_FORTRAN_DOUBLE_COMPLEX"
	cdef int RSB_NUMERICAL_TYPE_DEFAULT "RSB_NUMERICAL_TYPE_DEFAULT"
	cdef int RSB_NUMERICAL_TYPE_DEFAULT_INTEGER "RSB_NUMERICAL_TYPE_DEFAULT_INTEGER"
	cdef int RSB_NUMERICAL_TYPE_INVALID_TYPE "RSB_NUMERICAL_TYPE_INVALID_TYPE"
	cdef int RSB_NUMERICAL_TYPE_FIRST_BLAS "RSB_NUMERICAL_TYPE_FIRST_BLAS"
	cdef int RSB_NUMERICAL_TYPE_PREPROCESSOR_SYMBOLS "RSB_NUMERICAL_TYPE_PREPROCESSOR_SYMBOLS"
	cdef int RSB_BLAS_NUMERICAL_TYPE_PREPROCESSOR_SYMBOLS "RSB_BLAS_NUMERICAL_TYPE_PREPROCESSOR_SYMBOLS"
	cdef int RSB_MATRIX_STORAGE_DOUBLE_PRINTF_STRING "RSB_MATRIX_STORAGE_DOUBLE_PRINTF_STRING"
	cdef int RSB_MATRIX_STORAGE_FLOAT_PRINTF_STRING "RSB_MATRIX_STORAGE_FLOAT_PRINTF_STRING"
	cdef int RSB_MATRIX_STORAGE_FLOAT_COMPLEX_PRINTF_STRING "RSB_MATRIX_STORAGE_FLOAT_COMPLEX_PRINTF_STRING"
	cdef int RSB_MATRIX_STORAGE_DOUBLE_COMPLEX_PRINTF_STRING "RSB_MATRIX_STORAGE_DOUBLE_COMPLEX_PRINTF_STRING"
	cdef int RSB_TRANSPOSITIONS_PREPROCESSOR_SYMBOLS "RSB_TRANSPOSITIONS_PREPROCESSOR_SYMBOLS"
	cdef rsb_err_t rsb_strerror_r(rsb_err_t errval, rsb_char_t * buf, size_t buflen)
	cdef rsb_err_t rsb_perror(void_ptr stream, rsb_err_t errval)
	cdef struct rsb_initopts
	cdef rsb_err_t rsb_lib_init(rsb_opt_ptr  iop)
	cdef rsb_err_t rsb_lib_reinit(rsb_opt_ptr  iop)
	cdef rsb_err_t rsb_lib_set_opt_str(rsb_char_t* opnp, rsb_char_t* opvp)
	cdef rsb_err_t rsb_lib_set_opt(rsb_opt_t iof, cvoid_ptr iop)
	cdef rsb_err_t rsb_lib_get_opt(rsb_opt_t iof, void_ptr iop)
	cdef rsb_err_t rsb_lib_exit(rsb_opt_ptr  iop)
	cdef rsb_mtx_ptr  rsb_mtx_alloc_from_coo_begin(rsb_nnz_idx_t nnzA, rsb_type_t typecode, rsb_coo_idx_t nrA, rsb_coo_idx_t ncA, rsb_flags_t flagsA, rsb_err_t * errvalp)
	cdef rsb_err_t rsb_mtx_alloc_from_coo_end(rsb_mtx_ptr * mtxApp)
	cdef rsb_mtx_ptr  rsb_mtx_alloc_from_csr_const(cvoid_ptr VA, rsb_coo_idx_t * RP, rsb_coo_idx_t * JA, rsb_nnz_idx_t nnzA, rsb_type_t typecode, rsb_coo_idx_t nrA, rsb_coo_idx_t ncA, rsb_blk_idx_t brA, rsb_blk_idx_t bcA, rsb_flags_t flagsA, rsb_err_t * errvalp)
	cdef rsb_mtx_ptr  rsb_mtx_alloc_from_csc_const(cvoid_ptr VA, rsb_coo_idx_t * IA, rsb_coo_idx_t * CP, rsb_nnz_idx_t nnzA, rsb_type_t typecode, rsb_coo_idx_t nrA, rsb_coo_idx_t ncA, rsb_blk_idx_t brA, rsb_blk_idx_t bcA, rsb_flags_t flagsA, rsb_err_t * errvalp)
	cdef rsb_mtx_ptr  rsb_mtx_alloc_from_csr_inplace(void_ptr  VA, rsb_nnz_idx_t * RP, rsb_coo_idx_t * JA, rsb_nnz_idx_t nnzA, rsb_type_t typecode, rsb_coo_idx_t nrA, rsb_coo_idx_t ncA, rsb_blk_idx_t brA, rsb_blk_idx_t bcA, rsb_flags_t flagsA, rsb_err_t * errvalp)
	cdef rsb_mtx_ptr  rsb_mtx_alloc_from_coo_const(cvoid_ptr VA, rsb_coo_idx_t * IA, rsb_coo_idx_t * JA, rsb_nnz_idx_t nnzA, rsb_type_t typecode, rsb_coo_idx_t nrA, rsb_coo_idx_t ncA, rsb_blk_idx_t brA, rsb_blk_idx_t bcA, rsb_flags_t flagsA, rsb_err_t * errvalp)
	cdef rsb_mtx_ptr  rsb_mtx_alloc_from_coo_inplace(void_ptr VA, rsb_coo_idx_t * IA, rsb_coo_idx_t * JA, rsb_nnz_idx_t nnzA, rsb_type_t typecode, rsb_coo_idx_t nrA, rsb_coo_idx_t ncA, rsb_blk_idx_t brA, rsb_blk_idx_t bcA, rsb_flags_t flagsA, rsb_err_t * errvalp )
	cdef rsb_err_t rsb_mtx_clone(rsb_mtx_ptr * mtxBpp, rsb_type_t typecode, rsb_trans_t transA, cvoid_ptr alphap, rsb_mtx_ptr  mtxAp, rsb_flags_t flags)
	cdef rsb_mtx_ptr  rsb_mtx_free(rsb_mtx_ptr  mtxAp)
	cdef rsb_err_t rsb_mtx_get_nrm(rsb_mtx_ptr  mtxAp , void_ptr  Np, rsb_extff_t flags)
	cdef rsb_err_t rsb_mtx_get_vec(rsb_mtx_ptr  mtxAp , void_ptr  Dp, rsb_extff_t flags)
	cdef rsb_err_t rsb_mtx_rndr(rsb_char_t * filename, rsb_mtx_ptr mtxAp, rsb_coo_idx_t pmWidth, rsb_coo_idx_t pmHeight, rsb_marf_t rflags)
	cdef rsb_err_t rsb_file_mtx_rndr(void_ptr  pmp, rsb_char_t * filename, rsb_coo_idx_t pmlWidth, rsb_coo_idx_t pmWidth, rsb_coo_idx_t pmHeight, rsb_marf_t rflags)
	cdef rsb_err_t rsb_spmv(rsb_trans_t transA, cvoid_ptr alphap, rsb_mtx_ptr  mtxAp, cvoid_ptr  Xp, rsb_coo_idx_t incX, cvoid_ptr  betap, void_ptr  Yp, rsb_coo_idx_t incY)
	cdef rsb_err_t rsb_spmm(rsb_trans_t transA, cvoid_ptr  alphap, rsb_mtx_ptr  mtxAp, rsb_coo_idx_t nrhs, rsb_flags_t order, cvoid_ptr  Bp, rsb_nnz_idx_t ldB, cvoid_ptr  betap, void_ptr  Cp, rsb_nnz_idx_t ldC)
	cdef rsb_err_t rsb_spsv(rsb_trans_t transT, cvoid_ptr  alphap, rsb_mtx_ptr  mtxTp, cvoid_ptr  Xp, rsb_coo_idx_t incX, void_ptr  Yp, rsb_coo_idx_t incY)
	cdef rsb_err_t rsb_spsm(rsb_trans_t transT, cvoid_ptr  alphap, rsb_mtx_ptr  mtxTp, rsb_coo_idx_t nrhs, rsb_flags_t order, cvoid_ptr  betap, cvoid_ptr  Bp, rsb_nnz_idx_t ldB, void_ptr  Cp, rsb_nnz_idx_t ldC)
	cdef rsb_err_t rsb_mtx_add_to_dense(cvoid_ptr alphap, rsb_mtx_ptr  mtxAp, rsb_nnz_idx_t ldB, rsb_nnz_idx_t nrB, rsb_nnz_idx_t ncB, rsb_bool_t rowmajorB, void_ptr  Bp)
	cdef rsb_mtx_ptr  rsb_sppsp(rsb_type_t typecode, rsb_trans_t transA, cvoid_ptr alphap, rsb_mtx_ptr  mtxAp, rsb_trans_t transB, cvoid_ptr betap, rsb_mtx_ptr  mtxBp, rsb_err_t * errvalp)
	cdef rsb_mtx_ptr  rsb_spmsp(rsb_type_t typecode, rsb_trans_t transA, cvoid_ptr alphap, rsb_mtx_ptr  mtxAp, rsb_trans_t transB, cvoid_ptr betap, rsb_mtx_ptr  mtxBp, rsb_err_t * errvalp)
	cdef rsb_err_t rsb_spmsp_to_dense(rsb_type_t typecode, rsb_trans_t transA, cvoid_ptr alphap, rsb_mtx_ptr  mtxAp, rsb_trans_t transB, cvoid_ptr betap, rsb_mtx_ptr  mtxBp , rsb_nnz_idx_t ldC, rsb_nnz_idx_t nrC, rsb_nnz_idx_t ncC, rsb_bool_t rowmajorC, void_ptr Cp)
	cdef rsb_err_t rsb_mtx_switch_to_coo(rsb_mtx_ptr  mtxAp, void_ptr * VAp, rsb_coo_idx_t ** IAp, rsb_coo_idx_t ** JAp, rsb_flags_t flags)
	cdef rsb_err_t rsb_mtx_switch_to_csr(rsb_mtx_ptr  mtxAp, void_ptr * VAp, rsb_coo_idx_t ** IAp, rsb_coo_idx_t ** JAp, rsb_flags_t flags)
	cdef rsb_err_t rsb_mtx_get_coo(rsb_mtx_ptr  mtxAp, void_ptr  VA, rsb_coo_idx_t * IA, rsb_coo_idx_t * JA, rsb_flags_t flags )
	cdef rsb_err_t rsb_mtx_get_csr(rsb_type_t typecode, rsb_mtx_ptr  mtxAp, void_ptr  VA, rsb_nnz_idx_t * RP, rsb_coo_idx_t * JA, rsb_flags_t flags )
	cdef rsb_err_t rsb_mtx_get_rows_sparse(rsb_trans_t transA, cvoid_ptr  alphap, rsb_mtx_ptr  mtxAp, void_ptr  VA, rsb_coo_idx_t * IA, rsb_coo_idx_t * JA, rsb_coo_idx_t frA, rsb_coo_idx_t lrA, rsb_nnz_idx_t *rnzp, rsb_flags_t flags )
	cdef rsb_err_t rsb_mtx_get_coo_block(rsb_mtx_ptr  mtxAp, void_ptr  VA, rsb_coo_idx_t * IA, rsb_coo_idx_t * JA, rsb_coo_idx_t frA, rsb_coo_idx_t lrA, rsb_coo_idx_t fcA, rsb_coo_idx_t lcA, rsb_coo_idx_t * IREN, rsb_coo_idx_t * JREN, rsb_nnz_idx_t *rnzp, rsb_flags_t flags )
	cdef rsb_err_t rsb_mtx_get_info(rsb_mtx_ptr mtxAp, rsb_mif_t miflags, void_ptr  minfop)
	cdef rsb_err_t rsb_mtx_get_info_str(rsb_mtx_ptr mtxAp, rsb_char_t *mis, void_ptr  minfop, size_t buflen)
	cdef rsb_err_t rsb_mtx_upd_vals(rsb_mtx_ptr  mtxAp, rsb_elopf_t elop_flags, cvoid_ptr  omegap)
	cdef rsb_err_t rsb_mtx_get_prec(void_ptr opdp, rsb_mtx_ptr  mtxAp, rsb_precf_t prec_flags, cvoid_ptr ipdp)
	cdef rsb_err_t rsb_mtx_set_vals(rsb_mtx_ptr  mtxAp, cvoid_ptr  VA, rsb_coo_idx_t *IA, rsb_coo_idx_t *JA, rsb_nnz_idx_t nnz, rsb_flags_t flags)
	cdef rsb_err_t rsb_mtx_get_vals(rsb_mtx_ptr  mtxAp, void_ptr  VA, rsb_coo_idx_t *IA, rsb_coo_idx_t *JA, rsb_nnz_idx_t nnz, rsb_flags_t flags)
	cdef rsb_err_t rsb_tune_spmm(rsb_mtx_ptr * mtxOpp, rsb_real_t *sfp, rsb_int_t *tnp, rsb_int_t maxr, rsb_time_t maxt, rsb_trans_t transA, cvoid_ptr  alphap, rsb_mtx_ptr  mtxAp, rsb_coo_idx_t nrhs, rsb_flags_t order, cvoid_ptr  Bp, rsb_nnz_idx_t ldB, cvoid_ptr  betap, void_ptr  Cp, rsb_nnz_idx_t ldC)
	cdef rsb_err_t rsb_tune_spsm(rsb_mtx_ptr * mtxOpp, rsb_real_t *sfp, rsb_int_t *tnp, rsb_int_t maxr, rsb_time_t maxt, rsb_trans_t transA, cvoid_ptr  alphap, rsb_mtx_ptr  mtxAp, rsb_coo_idx_t nrhs, rsb_flags_t order, cvoid_ptr  Bp, rsb_nnz_idx_t ldB, cvoid_ptr  betap, void_ptr  Cp, rsb_nnz_idx_t ldC)
	cdef rsb_trans_t rsb_psblas_trans_to_rsb_trans(char psbtrans)
	cdef rsb_err_t rsb_file_mtx_save(rsb_mtx_ptr  mtxAp, rsb_char_t * filename)
	cdef rsb_mtx_ptr  rsb_file_mtx_load(rsb_char_t * filename, rsb_flags_t flagsA, rsb_type_t typecode, rsb_err_t *errvalp)
	cdef rsb_err_t rsb_file_vec_load(rsb_char_t * filename, rsb_type_t typecode, void_ptr  Yp, rsb_coo_idx_t *yvlp)
	cdef rsb_err_t rsb_file_vec_save(rsb_char_t * filename, rsb_type_t typecode, cvoid_ptr  Yp, rsb_coo_idx_t yvl)
	cdef rsb_err_t rsb_file_mtx_get_dims(rsb_char_t * filename, rsb_coo_idx_t* nrp, rsb_coo_idx_t *ncp, rsb_coo_idx_t *nzp, rsb_flags_t*flagsp)
	cdef rsb_err_t rsb_coo_sort(void_ptr VA, rsb_coo_idx_t * IA, rsb_coo_idx_t * JA, rsb_nnz_idx_t nnzA, rsb_coo_idx_t nrA, rsb_coo_idx_t ncA,  rsb_type_t typecode, rsb_flags_t flagsA )
	cdef rsb_err_t rsb_coo_cleanup(rsb_coo_idx_t* nnzp, void_ptr  VA, rsb_coo_idx_t* IA, rsb_coo_idx_t* JA, rsb_nnz_idx_t nnzA, rsb_coo_idx_t nrA, rsb_coo_idx_t ncA, rsb_type_t typecode, rsb_flags_t flagsA )
	cdef rsb_time_t rsb_time()
