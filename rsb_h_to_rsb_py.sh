#!/bin/bash
#
# Copyright (C) 2008-2017 Michele Martone
# 
# This file is part of librsb.
# 
# librsb is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# 
# librsb is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with librsb; see the file COPYING.
# If not, see <http://www.gnu.org/licenses/>.
SRCDIR=
if test $# = 0 ; then SRCDIR=. ; else SRCDIR="$1"; fi
IF=${SRCDIR}/rsb.h
TF=${SRCDIR}/rsb_types.h
cat << EOF
"""
librsb for python
Proof of concept, very limited interface code.
Author: Michele Martone
"""
cdef extern from "rsb.h":
EOF
BTS='s/\s\s*/ /g;' # blanks to space
D2N='s/^.define\>\s*//g;' # define to null
FTT='s/^\([^ 	()]*\)\s.*$/\1 "\1"/g;' # first token twice
PCT='s/^/ctypedef /g;' # prepend ctypedef  # FIXME: use this
PCD='s/^/cdef /g;' # prepend cdef
PCI='s/^/cdef int /g;' # prepend cdef int
EHC='s/^\s*,*\s*//g;' # erase heading commas
ETB='s/\s*$//g;' # erase trailing blanks
ETS='s/;$//g;' # erase trailing semicolon
C2C='s/\<const\>\s*//g;s/\<void\>\s*\*/void_ptr /g;s/struct rsb_mtx_t\s*\*/rsb_mtx_ptr /g;s/struct rsb_mtx_t\s*\*\s*\*/rsb_mtx_pptr /g;s/struct rsb_initopts\s*\*/rsb_opt_ptr /g;' # C to Cython
PAT='s/^/	/g;'
(
cat << EOF
ctypedef char* char_ptr "char*"
ctypedef char* const_char_ptr "const char*"
ctypedef void* void_ptr  "void*"
ctypedef void* cvoid_ptr  "const void*"
ctypedef void* rsb_mtx_ptr  "struct rsb_mtx_t*"
ctypedef void** rsb_mtx_pptr  "struct rsb_mtx_t**"
ctypedef void* rsb_mtx_cptr  "const struct rsb_mtx_t*"
ctypedef void* rsb_opt_ptr  "struct rsb_initopts*"

ctypedef int rsb_opt_t "enum rsb_opt_t"
ctypedef int rsb_mif_t "enum rsb_mif_t"
ctypedef int rsb_elopf_t "enum rsb_elopf_t"
ctypedef int rsb_extff_t "enum rsb_extff_t"
EOF
grep '\<typedef\>' ${IF} | sed 's/\/.*$//g;s/;//g;s/\s\s*/ /g;s/\s*$//g;s/\(^.*\) \([^ ]*$\)/\1 \2 "\2"/g;s/typedef/ctypedef/g;'
grep '^.define\>\s\s*RSB_[^\s()]*\s' ${IF} | sed "${BTS}${D2N}${FTT}${PCI}"
grep '^\([ ,]*\) *RSB_\(MIF\|ELOPF\|IO_WANT\|EXTF\)_' ${IF} | sed "${EHC}${FTT}${PCI}" 
grep '^.define\>\s\s*\(RSB_NUMERICAL_TYPE_\|RSB_TRANSPOSITION_N\|RSB_SYMMETRY_\)*\s' ${TF} | grep -v '\\$' | sed "${BTS}${D2N}${FTT}${PCI}"
grep '^\(rsb\|struct rsb\).*$' ${IF} | sed "${C2C}${PCD}${ETS}""s/(void)/()/g"
) | uniq | sed "${ETB}${PAT}s/enum //g;"
# MARF and PRECF are defs
