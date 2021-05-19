"""
librsb for python
Proof of concept, very limited interface code.
Author: Michele Martone
License: GPLv3+
"""
import math
import sys
import numpy as np
import scipy as sp
import rsb


def printf(format, *args):
    """
    Printf-like shorthand.
    """
    sys.stdout.write(format % args)


def bench(timeout, a, x, y):
    """
    Benchmark multiplication operation.
    :param timeout: benchmark time
    :param a: matrix
    :param x: right hand side vector
    :param y: result vector
    :return: a tuple with operation time, benchmark time, performed iterations
    """
    zero_alloc = True
    iterations = 0
    dt = -rsb.rsb_time()

    if zero_alloc:
        if (isinstance(a,rsb.rsb_matrix)):
            while dt + rsb.rsb_time() < timeout or iterations == 0:
                iterations = iterations + 1
                a._spmm(x,y) # This form avoids the copy of y
        else:
            while dt + rsb.rsb_time() < timeout or iterations == 0:
                iterations = iterations + 1
                # y += a._mul_multivector(x) # inefficient
                sp.sparse._sparsetools.csr_matvecs(a.shape[0], a.shape[1], x.shape[1], a.indptr, a.indices, a.data, x.ravel(), y.ravel())
    else:
        while dt + rsb.rsb_time() < timeout or iterations == 0:
            iterations = iterations + 1
            y += a * x  # Inefficient (result created repeatedly) see __mul__
    dt = dt + rsb.rsb_time()
    op_dt = dt / iterations
    return (op_dt, dt, iterations)


WANT_VERBOSE = 0
WANT_AUTOTUNE = 0
WANT_VERBOSE_TUNING = False
WANT_PSF = "csr"
WANT_NRHS = [1, 2, 3, 4, 5, 6, 7, 8]
WANT_NRA = [10, 30, 100, 300, 1000, 3000, 10000]


def bench_both(a, c, psf, order='C', nrhs=1):
    """
    Perform comparative benchmark: rsb vs csr.
    :param a: rsb matrix
    :param c: csr matrix
    :param psf: format string for matrix c
    :param nrhs: number of right-hand-side vectors
    """
    timeout = 0.2
    # timeout=2.0
    if WANT_AUTOTUNE:
        a.autotune(verbose=WANT_VERBOSE_TUNING)
    if WANT_VERBOSE:
        print("Benchmarking SPMV on matrix ", a)
    if WANT_VERBOSE:
        a._mini_self_print_test()
    x = np.ones([a.shape[1], nrhs], dtype=a.dtype, order=order)
    y = np.ones([a.shape[0], nrhs], dtype=a.dtype, order=order)
    nnz = a.nnz
    if WANT_VERBOSE:
        a.do_print()
        print("x=", x)
        print("y=", y)
        print("Benchmarking y<-A*x+y ... ")
    (psf_dt, dt, iterations) = bench(timeout, c, x, y)
    psf_mflops = (2 * nrhs * nnz) / (psf_dt * 1e6)
    if WANT_VERBOSE:
        print(
            "Done ",
            iterations,
            " ",
            psf,
            " SPMV iterations in ",
            dt,
            " s: ",
            psf_dt,
            "s per iteration, ",
            psf_mflops,
            " MFLOPS",
        )
    (rsb_dt, dt, iterations) = bench(timeout, a, x, y)
    rsb_mflops = (2 * nrhs * nnz) / (rsb_dt * 1e6)
    if WANT_VERBOSE:
        print(
            "Done ",
            iterations,
            " rsb SPMV iterations in ",
            dt,
            " s: ",
            rsb_dt,
            "s per iteration, ",
            rsb_mflops,
            " MFLOPS",
        )
    su = psf_dt / rsb_dt
    if WANT_VERBOSE:
        print("Speedup of RSB over ", psf, " is ", su, "x")
    printf(
        "PYRSB: nr: %d  nc: %d  nnz: %d  speedup: %.1e  nrhs: %d  order: %c"
        "  psf_mflops: %.2e  psf_dt: %.2e  rsb_mflops: %.2e  rsb_dt: %.2e  rsb_nsubm: %d\n",
        a.shape[0],
        a.shape[1],
        nnz,
        su,
        nrhs,
        order,
        psf_mflops,
        psf_dt,
        rsb_mflops,
        rsb_dt,
        a.nsubm(),
    )
    if WANT_VERBOSE:
        print("y=", y)


def bench_matrix(a, c):
    """
    Perform comparative benchmark: rsb vs csr.
    :param a: rsb matrix
    :param c: csr matrix
    """
    for nrhs in WANT_NRHS:
        bench_both(a, c, WANT_PSF, 'C', nrhs)
        bench_both(a, c, WANT_PSF, 'F', nrhs)
    del a
    del c


def bench_random_files():
    """
    Perform comparative benchmark on randomly generated matrices.
    """
    for nrA in WANT_NRA:
        ncA = nrA
        dnst = (math.sqrt(1.0 * nrA)) / nrA
        # print("# generating ",nrA,"x",ncA," with density ",dnst)
        printf("# generating %d x %d with with density %.1e\n", nrA, ncA, dnst)
        gt = -rsb.rsb_time()
        c = sp.sparse.rand(nrA, ncA, density=dnst, format=WANT_PSF, dtype=sp.double)
        gt = gt + rsb.rsb_time()
        (I, J, V) = sp.sparse.find(c)
        V = rsb.rsb_dtype(V)
        c = sp.sparse.csr_matrix((V, (I, J)), [nrA, ncA])
        ct = -rsb.rsb_time()
        a = rsb.rsb_matrix((V, (I, J)), [nrA, ncA])
        ct = ct + rsb.rsb_time()
        printf("# generated a matrix with %.1e nnz in %.1e s (%.1e nnz/s), converted to RSB in %.1e s\n",a.nnz,gt,a.nnz/gt,ct)
        bench_matrix(a, c)


def bench_file(filename):
    """
    Perform comparative benchmark on matrices loaded from Matrix Market files.
    :param filename: a Matrix Market file
    """
    print("# loading from file ", filename)
    lt = - rsb.rsb_time()
    a = rsb.rsb_matrix(bytes(filename, encoding="utf-8"),dtype=np.float64)
    lt = lt + rsb.rsb_time()
    printf("# loaded a matrix with %.1e nnz in %.1e s (%.1e nnz/s)\n",a.nnz,lt,a.nnz/lt)
    printf("# loaded as type %s (default is %s)\n", a.dtype, rsb.rsb_dtype)
    if not a._is_unsymmetric():
        print("# NOTE: loaded RSB matrix is NOT unsymmetric, but scipy will only perform unsymmetric SpMM")
    if a is not None:
        (I, J, V) = a.find()
        c = sp.sparse.csr_matrix((V, (I, J)))
        bench_matrix(a, c)


if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        bench_file(arg)
else:
    # bench_file("venkat50.mtx.gz")
    bench_random_files()
    # a.save("file.mtx")
rsb.rsb_lib_exit()
