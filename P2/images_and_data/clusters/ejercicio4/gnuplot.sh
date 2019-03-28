#!/bin/bash
gnuplot -persist -e "set output 'graficaSSE.eps' ; set encoding utf8; set xtics 1; set xrange [1:]; set terminal postscript eps color; set xlabel 'Número de Clusters'; set ylabel 'SSE'; plot 'datos.txt' using 2 t 'Relación k-SSE' w l"
