
ifeq "$(HOSTNAME)" "your_host"
#export EXTRA_CFLAGS=
#export RSBPATH=/opt/librsb-debug
#export RSBPATH=/opt/librsb-optimized
#export LIBRSB_CONFIG=$(RSBPATH)/bin/librsb-config
#export LIBRSB_CONFIG=$(RSBPATH)/bin/librsb-config
#export LIBRSB_CONFIG_PATH=$(RSBPATH)/bin
#export LIBRSB_CONFIG_PATH=$(RSBPATH)/bin
export LIBRSB_CONFIG_PATH=
else
endif
#export CXXLAGS=$(shell $(RSBPATH)/bin/librsb-config  --cflags ; python3-config  --cflags) 
#export LDFLAGS=$(shell $(LIBRSB_CONFIG) --ldflags ) -fopenmp 
#export LIBS=$(shell /opt/librsb-debug/bin/librsb-config   --ldflags ) 
export LIBRSB_CONFIG=$(LIBRSB_CONFIG_PATH)librsb-config
export CFLAGS=$(shell $(LIBRSB_CONFIG)  --cflags ; python3-config  --cflags ) $(EXTRA_CFLAGS)
export LDFLAGS=$(shell $(LIBRSB_CONFIG) --ldflags --extra_libs )
export RSB_INCDIR=$(shell $(LIBRSB_CONFIG) --prefix )/include

all: test

rsb: local-librsb

local-librsb-dl:
	wget 'https://sourceforge.net/projects/librsb/files/librsb-1.2.0.9.tar.gz/download' -O librsb-1.2.0.9.tar.gz

local-librsb-get:
	test -f librsb-1.2.0.9.tar.gz || make local-librsb-dl
	make md5check
	tar xvzf librsb-1.2.0.9.tar.gz

md5check:
	echo "f34b2968f354f7c33e1fbcffd3955ed7  librsb-1.2.0.9.tar.gz" > librsb-1.2.0.9.tar.gz.md5
	md5sum -c librsb-1.2.0.9.tar.gz.md5 || false

local-librsb: local-librsb-get
	cd librsb-1.2.0.9 && ./configure CFLAGS=-O3\ -march=native\ -mtune=native\ -fPIC\ -pipe        --prefix=`pwd`/../local --enable-shared --disable-c-examples && make -j4 && make install

local-librsb-debug: local-librsb-get
	cd librsb-1.2.0.9 && ./configure CFLAGS=-O0\ -fPIC\ -ggdb\ -pipe --prefix=`pwd`/../local --enable-shared && make -j4 && make install

lp: local-librsb-pyrsb
local-librsb-pyrsb:
	make test LIBRSB_CONFIG_PATH=`pwd`/local/bin/

all-local: local-librsb local-librsb-pyrsb

# python -c 'import rsb; import numpy; rsb.rsb_lib_init();a=rsb.rsb_matrix([11.0,22.0],[1,2],[1,2]);b=rsb.rsb_matrix([110.,220.],[1,2],[1,2]);x=numpy.array([110.,220,330]);y=numpy.array([0.,0.,0.]);b.spmv(x,y);print x,y;print(a>=b);print(a);print(a*b);del a; del b;rsb.rsb_lib_exit()'
test: rsb.so
	if which pytest-3 ; then PYTHONPATH=. pytest-3 ; fi
	python3 demo1.py
	python3 demo2.py
	python3 test.py A.mtx
	python3 test.py H.mtx
	python3 test.py
	python3 test.py -r2 -OC A.mtx
	python3 test.py -a -r1,2 A.mtx
	python3 test.py -aa -r1,2 A.mtx
	python3 test.py -aaa -r1,2 A.mtx
	python3 test.py -aaaa -r1,2 A.mtx

itest:
	rm -fR build/ rsb.egg-info/ dist/
	python3 setup.py sdist bdist_wheel
	ls dist
	pip3 install --user $(shell ls dist/*.whl)
	python3 demo1.py
	yes | pip3 uninstall rsb
	pip3 install --user $(shell ls dist/*.tar.gz)
	python3 demo1.py
	yes | pip3 uninstall rsb

rsb.so: rsb.o setup.py
	python3 setup.py build_ext -i

rsb.o: rsb.c

RSB2PY=./rsb_h_to_rsb_py.sh

# uncomment the following to enable regeneration of librsb.pxd creation script
# SRCDIR=$(HOME)/src/librsb-O0-DZ
#$(RSB2PY): $(SRCDIR)/scripts/rsb_h_to_rsb_py.sh
#	cp -p $< $@

# uncomment the following to enable regeneration of librsb.pxd 
# librsb.pxd: $(RSB2PY) $(SRCDIR)/rsb.h
# 	$(RSB2PY) $(SRCDIR) | grep -v RSB_WANT_LONG_IDX_TYPE > $@
	
rsb.c: rsb.pyx librsb.pxd
	cython3    -I$(RSB_INCDIR)  rsb.pyx
	#cython -3 -I$(RSB_INCDIR)  rsb.pyx

clean:
	rm -f *.o *.so *.c

e:
	vim +set\ number  $(RSB_INCDIR)/rsb.h +split\ $(RSB_INCDIR)/rsb_types.h +split\ rsb.pyx +split\ librsb.pxd 

b:	rsb.so
	make test | grep PYRSB: > pyrsb.dat
	gnuplot pyrsb.gp

dist:
	rm -fR pyrsb
	mkdir pyrsb/
	cp -v `svn ls` pyrsb/
	tar cvzf pyrsb.tar.gz pyrsb
	ls -l pyrsb.tar.gz
	tar xvzf pyrsb.tar.gz
	md5sum pyrsb.tar.gz > pyrsb.tar.gz.md5

signed-dist: dist
	gpg -sbv -u 1DBB555AEA359B8AAF0C6B88E0E669C8EF1258B8 -a pyrsb.tar.gz

cleanall:
	rm -fR demo1.py
	rm -fR demo2.py
	rm -fR EGG-INFO
	rm -fR build
	rm -fR dist
	rm -fR pyrsb.egg-info __pycache__

demo1.py demo2.py: README.md
	LSSNIPS= awk -f md2sh.awk < $<

# cython3 python3-dev

