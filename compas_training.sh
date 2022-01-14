#!/bin/bash

for seed in 0 1 2 3 4 5 6 7 8 9 10
do
#	python3 main_final.py --dataset compas --method naive --seed $seed --constraint eopp --epoch 25 --iteration 1 --gpu 2
#	python3 influence_function_test.py --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 2 --calc_option grad_V --target None
#	python3 influence_function_test.py --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 2 --calc_option s_test --target None
#	python3 influence_function_test.py --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 2 --calc_option influence --target None
	
	for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
	do
		python3 main_final.py --dataset compas --method naive_leave_k_out --seed $seed --constraint eopp --epoch 25 --iteration 1 --gpu 2 --k $k
		python3 main_final.py --dataset compas --method naive_leave_bottom_k_out --seed $seed --constraint eopp --epoch 25 --iteration 1 --gpu 2 --k $k
	done
done
