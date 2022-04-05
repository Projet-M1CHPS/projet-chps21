
set term png
set output "plot_bench.png"

set logscale y
set logscale x

set xlabel "Matrix size"
set ylabel "Time (s)"

set parametric

set arrow from 64000,graph(0,0) to 64000,graph(1,1) nohead

plot "blas_values.txt" using 1:2 with lines title "blas", "clblast_values.txt" using 1:2 with lines title "clblast"