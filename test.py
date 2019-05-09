"""
librsb for python
Proof of concept, very limited interface code.
Author: Michele Martone
License: GPLv3+
"""
import rsb
import math
import numpy as np
import scipy as sp
import sys

import sys
def printf(format, *args):
    sys.stdout.write(format % args)

def bench(timeout,a,x,y):
    iterations=0
    dt=-rsb.rsb_time()
    while dt+rsb.rsb_time() < timeout:
        iterations=iterations+1
        y=a*x # See __mul__
        # a.spmm(x,y) # This form avoids the copy of y.
    dt=dt+rsb.rsb_time()
    op_dt=dt/iterations
    return (op_dt,dt,iterations)

want_verbose=0
want_autotune=0
want_verbose_tuning=False
want_psf='csr'
want_nrhs = [ 1, 2, 3, 4, 5, 6, 7, 8 ]
want_nrA = [ 10, 30, 100, 300, 1000, 3000, 10000, 30000 ] 

def bench_both(a,c,psf,nrhs=1):
    timeout=0.2
    # timeout=2.0
    if want_autotune:
        a.autotune(verbose=want_verbose_tuning)
    if want_verbose:
    	print ("Benchmarking SPMV on matrix ",a)
    if want_verbose:
        print ("*")
        print (a)
        print ("*")
        print ("a:")
        print (a.find())
        print ("a's (1,1):")
        print (a.find_block(1,1,1,1))
        print ("a's tril")
        print (a.tril())
        print ("a's triu")
        print (a.triu())
        print (" ")
    x=np.ones([a.shape[1],nrhs],dtype=sp.double)
    y=np.ones([a.shape[0],nrhs],dtype=sp.double)
    nnz=a.nnz()
    if want_verbose:
        a.do_print()
        print("x=",x)
        print("y=",y)
        print("Benchmarking y<-A*x+y ... ")
    (psf_dt,dt,iterations)=bench(timeout,c,x,y)
    psf_mflops=(2*nrhs*nnz)/(psf_dt*1e6)
    if want_verbose:
    	print("Done ",iterations," ",psf," SPMV iterations in ",dt," s: ",psf_dt,"s per iteration, ",psf_mflops," MFLOPS")
    (rsb_dt,dt,iterations)=bench(timeout,a,x,y)
    rsb_mflops=(2*nrhs*nnz)/(rsb_dt*1e6)
    if want_verbose:
    	print("Done ",iterations," rsb SPMV iterations in ",dt," s: ",rsb_dt,"s per iteration, ",rsb_mflops," MFLOPS")
    su=psf_dt/rsb_dt
    if want_verbose:
    	print("Speedup of RSB over ",psf," is ",su,"x")
    #print("PYRSB:"," nr: ",a.shape[0]," nc: ",a.shape[1]," nnz: ",nnz," speedup: ",su," nrhs: ",nrhs," psf_mflops: ",psf_mflops," rsb_mflops: ",rsb_mflops,"")
    printf("PYRSB: nr: %d  nc: %d  nnz: %d  speedup: %.1e  nrhs: %d  psf_mflops: %.2e  rsb_mflops: %.2e\n",a.shape[0],a.shape[1],nnz,su,nrhs,psf_mflops,rsb_mflops)
    if want_verbose:
        print("y=",y)

#def bench_file(filename):

def bench_matrix(a,c):
    for nrhs in want_nrhs:
    	bench_both(a,c,want_psf,nrhs)
    del a;
    del c;

def bench_random_files():
    for nrA in want_nrA:
        ncA=nrA
        dnst=(math.sqrt(1.0*nrA))/nrA
        # print("# generating ",nrA,"x",ncA," with density ",dnst)
        printf("# generating %d x %d with with density %.1e\n",nrA,ncA,dnst)
        c=sp.sparse.rand(nrA,ncA,density=dnst,format=want_psf,dtype=sp.double)
        (I,J,V)=sp.sparse.find(c)
        a=rsb.rsb_matrix((V,(I,J)),[nrA,ncA])
        bench_matrix(a,c)

def bench_file(filename):
    print("# loading from file ",filename)
    a=rsb.rsb_file_mtx_load(filename)
    if a is not None:
        (I,J,V)=a.find()
        c=sp.sparse.csr_matrix((V,(I,J)))
        bench_matrix(a,c)

#bench_file("venkat50.mtx.gz")
bench_random_files()
# a.save("file.mtx")
rsb.rsb_lib_exit()
