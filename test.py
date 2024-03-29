"""
librsb for python
Proof of concept, very limited interface code.
Author: Michele Martone
License: GPLv3+
"""
import math
import sys
import getopt
import os, socket, datetime
import numpy as np
import scipy as sp
import rsb


WANT_ZERO_ALLOC = True
WANT_MAX_DUMP_NNZ = 16
WANT_VERBOSE = 0
WANT_AUTOTUNE = 0 # 0..
WANT_VERBOSE_TUNING = False
WANT_LIBRSB_STYLE_OUTPUT = False
WANT_PSF = "csr"
WANT_BOTH = True
WANT_NRHS = [1, 2, 4, 8]
WANT_ORDER = [ 'C', 'F' ]
WANT_NRA = [10, 30, 100, 300, 1000, 3000, 10000]
WANT_TYPES = [ 'S','D','C','Z' ]
WANT_TIMEOUT = 0.2
WANT_SYMMETRIZE = True
WANT_RENDER = False
TC2DT = {
            'S': np.float32,
            'D': np.float64,
            'C': np.complex64,
            'Z': np.complex128
        }
DT2TC = {
            np.float32: 'S',
            np.float64: 'D',
            np.complex64: 'C',
            np.complex128: 'Z'
        }
WANT_DTYPES = [ np.float32, np.float64, np.complex64, np.complex128 ]


def sprintf(format, *args):
    """
    Sprintf-like shorthand.
    """
    return format % args

def printf(format, *args):
    """
    Printf-like shorthand.
    """
    sys.stdout.write(sprintf(format % args))

def bench(timeout, a, x, y):
    """
    Benchmark multiplication operation.
    :param timeout: benchmark time
    :param a: matrix
    :param x: right hand side vector
    :param y: result vector
    :return: a tuple with min operation time, benchmark time, performed iterations
    """
    iterations = 0

    if timeout > 0.0:
        bench(0.0, a, x, y) # single op to warm-up caches

    op_dt = float('+inf')
    t0 = rsb.rsb_time()
    t1 = t0
    if WANT_ZERO_ALLOC:
        if (isinstance(a,rsb.rsb_matrix)):
            while t1 - t0 < timeout or iterations == 0:
                iterations = iterations + 1
                t2 = rsb.rsb_time()
                a._spmm(x,y) # This form avoids the copy of y
                t1 = rsb.rsb_time()
                op_dt = min(op_dt,t1-t2)
        else:
            while t1 - t0 < timeout or iterations == 0:
                iterations = iterations + 1
                t2 = rsb.rsb_time()
                # y += a._mul_multivector(x) # inefficient
                sp.sparse._sparsetools.csr_matvecs(a.shape[0], a.shape[1], x.shape[1], a.indptr, a.indices, a.data, x.ravel(), y.ravel())
                t1 = rsb.rsb_time()
                op_dt = min(op_dt,t1-t2)
    else:
        while t1 - t0 < timeout or iterations == 0:
            iterations = iterations + 1
            t2 = rsb.rsb_time()
            y += a * x  # Inefficient (result created repeatedly) see __mul__
            t1 = rsb.rsb_time()
            op_dt = min(op_dt,t1-t2)
    bt = rsb.rsb_time() - t0
    return (op_dt, bt, iterations)


def print_perf_record(pr,beg="",end="\n",fields=False,omit_samples_field=True):
    printf("%spr:    ",beg)
    for field in list(pr):
        if fields:
            if omit_samples_field and field.endswith('_samples'):
                printf("/")
            else:
                printf("%s:",field)
        value = pr[field]
        if field in [ 'BPNZ', 'AT_BPNZ' ]:
            printf("%.2f",value)
        elif type(value) is int:
            printf("%d",value)
        elif type(value) is float:
            printf("%.2e",value)
        elif type(value) is str:
            printf("%s",value)
        else:
            printf("?")
        printf(" ")
    printf("%s",end)


def bench_record(a, psf, brdict, order, nrhs, rsb_dt, psf_dt, rsb_at_dt=None, psf_at_dt=None):
    """
    Print benchmark record line
    """
    nnz = a.nnz
    su = psf_dt / rsb_dt
    if rsb_at_dt == None:
        rsb_at_dt = rsb_dt
    else:
        rsb_at_dt = min(rsb_dt,rsb_at_dt) # noise plus very close values may lead to (rsb_dt<rsb_at_dt)
    if psf_at_dt == None:
        psf_at_dt = psf_dt
    else:
        psf_at_dt = min(psf_dt,psf_at_dt)
        psf_dt = psf_at_dt # scipy.sparse has no autotuning -- we just take best time
    if WANT_BOTH:
        psf_mflops = (2 * nrhs * nnz) / (psf_dt * 1e6)
        psf_at_mflops = (2 * nrhs * nnz) / (psf_at_dt * 1e6)
    else:
        psf_mflops = 0.0
        psf_at_mflops = 0.0
    rsb_mflops = (2 * nrhs * nnz) / (rsb_dt * 1e6)
    rsb_at_mflops = (2 * nrhs * nnz) / (rsb_at_dt * 1e6)
    if WANT_VERBOSE:
        print("Speedup of RSB over ", psf, " is ", su, "x")
    if WANT_LIBRSB_STYLE_OUTPUT:
        # in the style of librsb's output (unfinished)
        SEP = " "
        BESTCODE = order # FIXME: note order shouldn't be here
        if rsb_dt > psf_dt:
            BESTCODE += ":P"
        else:
            BESTCODE += ":R"
        TYPE = a._get_typechar()
        SYM = a._get_symchar()
        TRANS = "N"
        MTX = brdict['mtxname']
        NT = rsb._get_rsb_threads()
        AT_NT = NT # AT-NT
        AT_SPS_NT = NT # AT-SPS-NT
        BPNZ = brdict['bpnz']
        AT_BPNZ = a._idx_bpnz()
        NSUBM = brdict['nsubm']
        AT_NSUBM = a.nsubm()
        RSBBEST_MFLOPS = max(rsb_mflops,rsb_at_mflops)
        OPTIME = rsb_dt
        SPS_OPTIME = psf_dt
        AT_SPS_OPTIME = psf_at_dt
        AT_OPTIME = rsb_at_dt
        AT_TIME = brdict['at_time']
        RWminBW_GBps = 1.0 # FIXME (shall be read + write traffic of operands and matrix)
        AT_MS = 0.0 # fixed to 0 here
        CMFLOPS = 2*(nnz/1e6)*nrhs
        if not a._is_unsymmetric():
            CMFLOPS *= 2
        if a._is_complex():
            CMFLOPS *= 4
        el_sizes = {
            'D': 8,
            'S': 4,
            'C': 8,
            'Z': 16
        }
        mt = 0.0 + a._total_size + el_sizes[a._get_typechar()]*(a.shape[0]+a.shape[1])*nrhs # minimal traffic
        CB_bpf = ( 1.0 / ( 1e6 * CMFLOPS ) ) * mt
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
    if WANT_LIBRSB_STYLE_OUTPUT:
        br = {
                'BESTCODE' :BESTCODE,
                'MTX' : MTX,
                'NR' : a.shape[0],
                'NC' : a.shape[1],
                'NNZ' : nnz,
                'NRHS' : nrhs,
                'TYPE' : TYPE,
                'SYM' : SYM,
                'TRANS' : TRANS,
                'NT' : NT,
                'AT_NT' : AT_NT,
                'AT_SPS_NT' : AT_SPS_NT,
                'BPNZ' : BPNZ,
                'AT_BPNZ' : AT_BPNZ,
                'NSUBM' : NSUBM,
                'AT_NSUBM': AT_NSUBM,
                'RSBBEST_MFLOPS' : RSBBEST_MFLOPS,
                'OPTIME' : OPTIME,
                'SPS_OPTIME' : SPS_OPTIME,
                'AT_OPTIME' : AT_OPTIME,
                'AT_SPS_OPTIME' : AT_SPS_OPTIME,
                'AT_TIME' : AT_TIME,
                'RWminBW_GBps' : RWminBW_GBps,
                'CB_bpf' : CB_bpf,
                'AT_MS' : AT_MS,
                'CMFLOPS' : CMFLOPS,
                }
        print_perf_record(br)
    else:
        br = { }
    return br


def bench_both(a, c, psf, brdict, order='C', nrhs=1):
    """
    Perform comparative benchmark: rsb vs csr.
    :param a: rsb matrix
    :param c: csr matrix
    :param psf: format string for matrix c
    :param nrhs: number of right-hand-side vectors
    """
    timeout = WANT_TIMEOUT
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
    if WANT_BOTH:
        (psf_dt, bt, iterations) = bench(timeout, c, x, y)
        if WANT_VERBOSE:
            print(
                "Done ",
                iterations,
                " ",
                psf,
                " SPMV iterations in ",
                bt,
                " s: ",
                psf_dt,
                "s per iteration, ",
            )
    else:
        (psf_dt, bt, iterations) = (0.0, 0.0, 0)
    (rsb_dt, bt, iterations) = bench(timeout, a, x, y)
    if WANT_VERBOSE:
        print(
            "Done ",
            iterations,
            " rsb SPMV iterations in ",
            bt,
            " s: ",
            rsb_dt,
            "s per iteration, ",
        )
    return [rsb_dt,psf_dt]


def bench_matrix(a, c, mtxname):
    """
    Perform comparative benchmark: rsb vs csr.
    :param a: rsb matrix
    :param c: csr matrix
    """
    brdict = {
        'mtxname': mtxname,
        'at_time': 0.0,
        'nsubm': a.nsubm(),
        'bpnz': a._idx_bpnz()
    }
    #tmax = 2
    #tmax = 0.1
    #tmax = -3
    tmax = 0
    bd = dict()
    psf = WANT_PSF
    if WANT_RENDER:
        filename = sprintf("%s-%c.eps",mtxname,DT2TC[a.dtype])
        a.render(filename)
    for nrhs in WANT_NRHS:
        bd[nrhs] = dict()
    if WANT_AUTOTUNE == 0:
        for nrhs in WANT_NRHS:
            for order in WANT_ORDER:
                (rsb_dt,psf_dt) = bench_both(a, c, psf, brdict, order, nrhs)
                bd[nrhs][order] = bench_record(a, psf, brdict, order, nrhs, rsb_dt, psf_dt)
    elif WANT_AUTOTUNE == 1:
        o = a.copy()
        if WANT_VERBOSE:
            print("Will autotune matrix for SpMV    ", a)
        at_time = rsb.rsb_time()
        o.autotune(verbose=WANT_VERBOSE_TUNING,tmax=tmax)
        brdict['at_time'] = rsb.rsb_time() - at_time
        if WANT_RENDER:
            filename = sprintf("%s-%c-tuned.eps",mtxname,DT2TC[o.dtype])
            o.render(filename)
        for nrhs in WANT_NRHS:
            for order in WANT_ORDER:
                (rsb_dt,psf_dt) = bench_both(a, c, psf, brdict, order, nrhs)
                (rsb_at_dt,psf_at_dt) = bench_both(o, c, psf, brdict, order, nrhs)
                bd[nrhs][order] = bench_record(o, psf, brdict, order, nrhs, rsb_dt, psf_dt, rsb_at_dt, psf_at_dt)
        del o
    elif WANT_AUTOTUNE == 2:
        for nrhs in WANT_NRHS:
            for order in WANT_ORDER:
                if WANT_VERBOSE:
                    print("Will autotune one matrix instance for different specific SpMM    ", a)
                (rsb_dt,psf_dt) = bench_both(a, c, psf, brdict, order, nrhs)
                at_time = rsb.rsb_time()
                a.autotune(verbose=WANT_VERBOSE_TUNING,nrhs=nrhs,order=order,tmax=tmax)
                brdict['at_time'] = rsb.rsb_time() - at_time
                if WANT_RENDER:
                    filename = sprintf("%s-%c-tuned-%c-%d.eps",mtxname,DT2TC[a.dtype],order,nrhs)
                    a.render(filename)
                (rsb_at_dt,psf_at_dt) = bench_both(a, c, psf, brdict, order, nrhs)
                bd[nrhs][order] = bench_record(a, psf, brdict, order, nrhs, rsb_dt, psf_dt, rsb_at_dt, psf_at_dt)
    elif WANT_AUTOTUNE >= 3:
        for nrhs in WANT_NRHS:
            for order in WANT_ORDER:
                (rsb_dt,psf_dt) = bench_both(a, c, psf, brdict, order, nrhs)
                o = a.copy()
                if WANT_VERBOSE:
                    print("Will autotune copies of starting matrix for specific SpMM    ", a)
                at_time = rsb.rsb_time()
                for i in range(2,+WANT_AUTOTUNE):
                    o.autotune(verbose=WANT_VERBOSE_TUNING,nrhs=nrhs,order=order,tmax=tmax)
                brdict['at_time'] = rsb.rsb_time() - at_time
                if WANT_RENDER:
                    filename = sprintf("%s-%c-tuned-%c-%d.eps",mtxname,DT2TC[o.dtype],order,nrhs)
                    o.render(filename)
                (rsb_at_dt,psf_at_dt) = bench_both(o, c, psf, brdict, order, nrhs)
                bd[nrhs][order] = bench_record(o, psf, brdict, order, nrhs, rsb_dt, psf_dt, rsb_at_dt, psf_at_dt)
                del o
    del a
    del c
    return bd

def dict_sum_init(d,k):
    d[k] = 0.0
    d[k+'_samples'] = 0

def dict_sum_update(d,k,v,w=1):
    d[k] += v
    d[k+'_samples'] += w

def dict_stat_merge(d,s):
    """
    Run on list() keys to average samples
    """
    if s != None:
        for k in list(s):
            if not k.endswith('_samples'):
                if not k in list(d):
                    dict_sum_init(d,k)
                dict_sum_update(d,k,s[k],w=s[k+'_samples'])

def dict_sum_average_all(d):
    """
    Run on list() keys to average samples
    """
    for k in list(d):
        if not k.endswith('_samples'):
            if d[k+'_samples'] == 0:
                del(d[k+'_samples'])
                del(d[k])
            else:
                d[k] /= d[k+'_samples']

def derived_bench_stats(bd):
    """
    Print derived benchmark data statistics
    """
    ot_keys = ['OPTIME']
    if WANT_AUTOTUNE > 0:
        ot_keys += ['AT_OPTIME']
    if WANT_BOTH > 0:
        ot_keys += ['SPS_OPTIME']
    if not WANT_LIBRSB_STYLE_OUTPUT:
        return
    bs = dict()
    dict_sum_init(bs,'speedup_autotuned_over_non_tuned')
    dict_sum_init(bs,'speedup_autotuned_over_scipy')
    dict_sum_init(bs,'speedup_non_tuned_over_scipy')
    # note we skip amortization stats
    if len(WANT_ORDER) == 2:
        or0,or1 = ( WANT_ORDER[0], WANT_ORDER[1] )
        for ot_key in ot_keys:
            ouk = sprintf("order_%s_speedup_%c_over_%c",ot_key,or0,or1);
            dict_sum_init(bs,ouk)
            for nrhs in WANT_NRHS:
                if nrhs != 1:
                    dr0 = bd[nrhs][or0]
                    dr1 = bd[nrhs][or1]
                    drx = bd[nrhs][or1].copy()
                    del(drx[ot_key])
                    iuk = sprintf("order_%s_speedup_%c_over_%c_%d_rhs",ot_key,or0,or1,nrhs);
                    beg = sprintf("pyrsb:%s:",iuk);
                    end = sprintf(" %.2f\n",dr1[ot_key]/dr0[ot_key])
                    print_perf_record(drx,beg,end)
                    dict_sum_init(bs,iuk)
                    dict_sum_update(bs,iuk,dr1[ot_key]/dr0[ot_key])
                    dict_sum_update(bs,ouk,dr1[ot_key]/dr0[ot_key])
                    del(dr0,dr1,drx)
    if len(WANT_NRHS) >= 2 and WANT_NRHS[0] == 1:
        for ot_key in ot_keys:
            auk = sprintf("rhs_%s_speedup_over_1_rhs",ot_key)
            dict_sum_init(bs,auk)
            for nrhs in WANT_NRHS:
                if nrhs != 1:
                    ouk = sprintf("rhs_%d_%s_speedup_over_1_rhs",nrhs,ot_key)
                    dict_sum_init(bs,ouk)
                    for order in WANT_ORDER:
                        dr1 = bd[  1 ][order]
                        drn = bd[nrhs][order]
                        drx = bd[nrhs][order].copy()
                        del(drx[ot_key])
                        iuk = sprintf("rhs_%d_%s_speedup_over_1_rhs_%c_order",nrhs,ot_key,order)
                        beg = sprintf("pyrsb:%s:",iuk)
                        dict_sum_init(bs,iuk)
                        end = sprintf(" %.2f\n",nrhs*dr1[ot_key]/(drn[ot_key]))
                        dict_sum_update(bs,iuk,nrhs*dr1[ot_key]/(drn[ot_key]))
                        dict_sum_update(bs,ouk,nrhs*dr1[ot_key]/(drn[ot_key]))
                        dict_sum_update(bs,auk,nrhs*dr1[ot_key]/(drn[ot_key]))
                        print_perf_record(drx,beg,end)
                        del(dr1,drx,drn)
    for order in WANT_ORDER:
        for nrhs in WANT_NRHS:
            dr = bd[nrhs][order]
            if WANT_AUTOTUNE > 0:
                beg = sprintf("pyrsb:speedup_autotuned_over_non_tuned:");
                end = sprintf(" %.2f\n",dr['OPTIME']/dr['AT_OPTIME'])
                print_perf_record(dr,beg,end)
                dict_sum_update(bs,'speedup_autotuned_over_non_tuned',dr['OPTIME']/dr['AT_OPTIME'])
                if WANT_BOTH:
                    beg = sprintf("pyrsb:speedup_autotuned_over_scipy:");
                    end = sprintf(" %.2f\n",dr['SPS_OPTIME']/dr['AT_OPTIME'])
                    print_perf_record(dr,beg,end)
                    dict_sum_update(bs,'speedup_autotuned_over_scipy',dr['SPS_OPTIME']/dr['AT_OPTIME'])
                    beg = sprintf("pyrsb:amortize_tuning_over_scipy:");
                    if dr['SPS_OPTIME'] > dr['AT_OPTIME']:
                        end = sprintf(" %.2f\n",dr['AT_TIME']/(dr['SPS_OPTIME']-dr['AT_OPTIME']))
                    else:
                        end = sprintf(" %f\n",float('+inf'))
                    print_perf_record(dr,beg,end)
                beg = sprintf("pyrsb:amortize_tuning_over_untuned_rsb:");
                if dr['OPTIME'] > dr['AT_OPTIME']:
                    end = sprintf(" %.2f\n",dr['AT_TIME']/(dr['OPTIME']-dr['AT_OPTIME']))
                else:
                    end = sprintf(" %f\n",float('+inf'))
                print_perf_record(dr,beg,end)
            if WANT_BOTH:
                beg = sprintf("pyrsb:speedup_non_tuned_over_scipy:");
                end = sprintf(" %.2f\n",dr['SPS_OPTIME']/dr['OPTIME'])
                print_perf_record(dr,beg,end)
                dict_sum_update(bs,'speedup_non_tuned_over_scipy',dr['SPS_OPTIME']/dr['OPTIME'])
    return bs

def bench_random_matrices():
    """
    Perform comparative benchmark on randomly generated matrices.
    """
    for dtype in WANT_DTYPES:
        for nrA in WANT_NRA:
            ncA = nrA
            dnst = (math.sqrt(1.0 * nrA)) / nrA
            # print("# generating ",nrA,"x",ncA," with density ",dnst)
            printf("# generating %d x %d with with density %.1e\n", nrA, ncA, dnst)
            gt = -rsb.rsb_time()
            c = sp.sparse.rand(nrA, ncA, density=dnst, format=WANT_PSF, dtype=rsb.rsb_dtype)
            gt = gt + rsb.rsb_time()
            (I, J, V) = sp.sparse.find(c)
            V = dtype(V)
            c = sp.sparse.csr_matrix((V, (I, J)), [nrA, ncA])
            ct = -rsb.rsb_time()
            a = rsb.rsb_matrix((V, (I, J)), [nrA, ncA], dtype=dtype)
            ct = ct + rsb.rsb_time()
            printf("# generated a matrix with %.1e nnz in %.1e s (%.1e nnz/s), converted to RSB in %.1e s\n",a.nnz,gt,a.nnz/gt,ct)
            bd = bench_matrix(a, c, "random")
            derived_bench_stats(bd)


def bench_file(filename):
    """
    Perform comparative benchmark on matrices loaded from Matrix Market files.
    :param filename: a Matrix Market file
    """
    bs = dict()
    for dtype in WANT_DTYPES:
    	print("# loading matrix from file ", filename)
    	lt = - rsb.rsb_time()
    	a = rsb.rsb_matrix(filename,dtype=dtype)
    	lt = lt + rsb.rsb_time()
    	printf("# loaded a matrix with %.1e nnz in %.1e s (%.1e nnz/s)\n",a.nnz,lt,a.nnz/lt)
    	printf("# loaded as type %s/%c (default is %s/%c)\n", a.dtype, DT2TC[a.dtype], rsb.rsb_dtype, DT2TC[rsb.rsb_dtype])
    	if not a._is_unsymmetric():
    	    if WANT_SYMMETRIZE:
    	        print("# NOTE: loaded RSB matrix is NOT unsymmetric: expanding symmetry to cope with csr.")
    	        (I, J, V) = a.find()
    	        a = rsb.rsb_matrix((np.append(V,V), (np.append(I,J), np.append(J,I))), a.shape, dtype=dtype)
    	    else:
    	        print("# NOTE: loaded RSB matrix is NOT unsymmetric, but scipy will only perform unsymmetric SpMM")
    	if a is not None:
    	    (I, J, V) = a.find()
    	    c = sp.sparse.csr_matrix((V, (I, J)))
    	    ( mtxname, _ ) = os.path.splitext(os.path.basename(filename))
    	    ( mtxname, _ ) = os.path.splitext(mtxname)
    	    bd = bench_matrix(a, c, mtxname)
    	    dict_stat_merge(bs, derived_bench_stats(bd))
    bsc = bs.copy()
    dict_sum_average_all(bsc)
    print_perf_record(bsc,beg="pyrsb:speedups:",fields=True)
    return bs

def elapsed_time():
    (_,_,_,_,elpt) = os.times()
    return elpt

print ("# PyRSB benchmark on ", socket.gethostname(), " ", datetime.datetime.now().ctime() )
try:
    opts,args = getopt.gnu_getopt(sys.argv[1:],"a:b:lpr:u:AO:RST:")
except getopt.GetoptError:
    sys.exit(1)
for o,a in opts:
    if o == '-a':
        WANT_AUTOTUNE = int(a)
    if o == '-b':
        WANT_TIMEOUT = float(a)
    if o == '-l':
        WANT_LIBRSB_STYLE_OUTPUT = True
    if o == '-p':
        WANT_RENDER = True
    if o == '-r':
        WANT_NRHS = list(map(int,a.split(',')))
    if o == '-u':
        WANT_NRA = list(map(int,a.split(',')))
    if o == '-A':
        WANT_ZERO_ALLOC = False
    if o == '-O':
        WANT_ORDER = list(map(lambda c : c.upper(),a.split(',')) )
    if o == '-R':
        WANT_BOTH = False
    if o == '-S':
        WANT_SYMMETRIZE = False
    if o == '-T':
        WANT_TYPES = list(a)
        WANT_DTYPES = list(map(lambda c : TC2DT[c.upper()],WANT_TYPES))
if len(opts) == 0:
    print ("# no custom options specified: using defaults")
if len(opts) >= 1:
    print ("# autotune:", WANT_AUTOTUNE )
    print ("# nrhs:", WANT_NRHS )
    print ("# order:", WANT_ORDER )
    print ("# librsb output:", WANT_LIBRSB_STYLE_OUTPUT )
    print ("# types:", WANT_TYPES )
    print ("# dtypes:", WANT_DTYPES )
    print ("# dims (if gen random):", WANT_NRA )
    print ("# bench timeout:", WANT_TIMEOUT )
    print ("# operands alloc:", not WANT_ZERO_ALLOC)
    print ("# want RSB and scipy.sparse:", WANT_BOTH)
    print ("# render:", WANT_RENDER)
    print ("# symmetrize:", WANT_SYMMETRIZE )
if len(args) > 0:
    bs = dict()
    if len(args):
        printf("# will load matrix files:")
        for arg in args[0:]:
            printf(" %s", arg)
        printf("\n")
    elt0 = elapsed_time()
    if True:
        # warm up threads
        u = rsb.rsb_matrix(np.ones(shape=(1000,1000)))
        u.autotune(verbose=False,nrhs=1,tmax=0.0001)
        del u
    for arg in args[0:]:
        bd = bench_file(arg)
        dict_stat_merge(bs, bd)
        elt1 = elapsed_time()
        printf("# time elapsed: %.1f s\n", elt1-elt0)
    dict_sum_average_all(bs)
    print_perf_record(bs,beg="pyrsb:speedups:",fields=True)
else:
    # bench_file("venkat50.mtx.gz")
    bench_random_matrices()
    # a.save("file.mtx")
rsb.rsb_lib_exit()
