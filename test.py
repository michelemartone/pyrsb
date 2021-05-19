"""
librsb for python
Proof of concept, very limited interface code.
Author: Michele Martone
License: GPLv3+
"""
import math
import sys
import os
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


WANT_MAX_DUMP_NNZ = 16
WANT_VERBOSE = 0
WANT_AUTOTUNE = 0 # 0..
WANT_VERBOSE_TUNING = False
WANT_PSF = "csr"
WANT_NRHS = [1, 2, 3, 4, 5, 6, 7, 8]
WANT_ORDER = [ 'C', 'F' ]
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
    if WANT_VERBOSE:
        print("Benchmarking SPMV on matrix ", a)
    x = np.ones([a.shape[1], nrhs], dtype=a.dtype, order=order)
    y = np.ones([a.shape[0], nrhs], dtype=a.dtype, order=order)
    nnz = a.nnz
    if WANT_VERBOSE and nnz <= WANT_MAX_DUMP_NNZ:
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

    if False:
        # in the style of librsb's output (unfinished)
        SEP = " "
        if rsb_dt > psf_dt:
            BESTCODE = order+":P" # FIXME: note order shouldn't be here
        else:
            BESTCODE = order+":R" # FIXME: note order shouldn't be here
        TYPE = "D" # FIXME
        if a._is_unsymmetric():
            SYM = "G" # FIXME
        else:
            SYM = "S" # FIXME
        TRANS = "N"
        MTX = "A" # FIXME: matrix name
        NT0 = 1 # FIXME: threads
        if os.environ.get("OMP_NUM_THREADS") is not None:
            NT0 = int(os.environ.get("OMP_NUM_THREADS"))
            # FIXME : shall use RSB_IO_WANT_EXECUTING_THREADS instead
        NT1 = NT0 # AT-NT
        NT2 = NT0 # AT-SPS-NT
        BPNZ = -1 # FIXME: see RSB_MIF_INDEX_STORAGE_IN_BYTES_PER_NNZ__TO__RSB_REAL_T
        AT_BPNZ = -1 # FIXME: see RSB_MIF_INDEX_STORAGE_IN_BYTES_PER_NNZ__TO__RSB_REAL_T
        NSUBM = a.nsubm() # FIXME
        AT_NSUBM = a.nsubm()
        RSBBEST_MFLOPS = rsb_mflops # FIXME: differentiate tuned from untuned
        OPTIME = rsb_dt # FIXME: differentiate tuned from untuned
        SPS_OPTIME = psf_dt
        AT_SPS_OPTIME = psf_dt
        AT_OPTIME = rsb_dt # FIXME: differentiate tuned from untuned
        AT_TIME = 0 # FIXME
        RWminBW_GBps = 1 # FIXME
        CB_bpf = 1 # FIXME
        AT_MS = 0 # FIXME: merge/split
        CMFLOPS = 2*(a.shape[0]/1e6)*a.shape[1]*nrhs
        if False: # FIXME: on complex
            CMFLOPS *= 4
        printf(
                "pr:    %s"
                "%s%s"
                "%s%d"
                "%s%d"
                "%s%d"
                "%s%d"
                "%s%c"
                "%s%c"
                "%s%c"
                "%s%d"
                "%s%d"
                "%s%d"
                "%s%.2e"
                "%s%.2e"
                "%s%d"
                "%s%d"
                "%s%.2e"
                "%s%.2e"
                "%s%.2e"
                "%s%.2e"
                "%s%.2e"
                "%s%.2e"
                "%s%.2e"
                "%s%.2e"
                "%s%.2e"
                "%s%.2e"
            "\n",
            BESTCODE,
            SEP, MTX,
            SEP, a.shape[0],
            SEP, a.shape[1],
            SEP, nnz,
            SEP, nrhs,
            SEP, TYPE,
            SEP, SYM,
            SEP, TRANS,
            SEP, NT0,
            SEP, NT1,
            SEP, NT2,
            SEP, BPNZ,
            SEP, AT_BPNZ,
            SEP, NSUBM,
            SEP, AT_NSUBM,
            SEP, RSBBEST_MFLOPS,
            SEP, OPTIME,
            SEP, SPS_OPTIME,
            SEP, AT_OPTIME,
            SEP, AT_SPS_OPTIME,
            SEP, AT_TIME,
            SEP, RWminBW_GBps,
            SEP, CB_bpf,
            SEP, AT_MS,
            SEP, CMFLOPS
        )
    else:
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
    if WANT_VERBOSE and nnz <= WANT_MAX_DUMP_NNZ:
        print("y=", y)


def bench_matrix(a, c):
    """
    Perform comparative benchmark: rsb vs csr.
    :param a: rsb matrix
    :param c: csr matrix
    """
    if WANT_AUTOTUNE == 0:
        for nrhs in WANT_NRHS:
            for order in WANT_ORDER:
                bench_both(a, c, WANT_PSF, order, nrhs)
    elif WANT_AUTOTUNE == 1:
        o = a.copy()
        if WANT_VERBOSE:
            print("Will autotune matrix for SpMV    ", a)
        o.autotune(verbose=WANT_VERBOSE_TUNING)
        for nrhs in WANT_NRHS:
            for order in WANT_ORDER:
                 bench_both(o, c, WANT_PSF, order, nrhs)
        del o
    elif WANT_AUTOTUNE == 2:
        for nrhs in WANT_NRHS:
            for order in WANT_ORDER:
                if WANT_VERBOSE:
                    print("Will autotune one matrix instance for different specific SpMM    ", a)
                a.autotune(verbose=WANT_VERBOSE_TUNING,nrhs=nrhs,order=ord(order))
                bench_both(a, c, WANT_PSF, order, nrhs)
    elif WANT_AUTOTUNE >= 3:
        for nrhs in WANT_NRHS:
            for order in WANT_ORDER:
                o = a.copy()
                if WANT_VERBOSE:
                    print("Will autotune copies of starting matrix for specific SpMM    ", a)
                for i in range(2,+WANT_AUTOTUNE):
                    o.autotune(verbose=WANT_VERBOSE_TUNING,nrhs=nrhs,order=ord(order))
                bench_both(o, c, WANT_PSF, order, nrhs)
                del o
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
