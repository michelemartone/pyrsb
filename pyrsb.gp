set terminal postscript eps enhanced color fontscale 1.3
set output "pyrsb.eps" 
set logscale x 2
unset logscale y 

#set xzeroaxis
set key right Right
#set xzeroaxis 
#set yzeroaxis 
#set logscale x 2
#unset logscale y 
#set logscale y 2
#set mxtics 1
#set xtics autofreq 2
unset xtics 
#set xtics norotate
#set xtics auto
set xtics format "%4.0f"
#set title "\_"
unset object
unset xlabel
set xlabel "matrix size"

unset style 
unset y2label
unset ylabel
unset ytics
set ytics
#set logscale y 2
set ylabel 'speedup wrt scipy.sparse.csr\_matrix'
#set object 1 rect from 1,1,0 to 2,2,0 front lw 1.0 fillcolor rgb "red" fillstyle empty border -1
plot  1, 'pyrsb.dat' using 3:9 title 'speedup' with points
