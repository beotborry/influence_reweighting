#!/bin/bash

for seed in 0 1 2 3 4 5 6 7 8 9
do
	python3 main_final.py --sen_attr sex --dataset compas --method naive --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 2
	python3 influence_function_test.py --option fair --sen_attr sex --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 2 --calc_option grad_V --target None
	python3 influence_function_test.py --option fair --sen_attr sex --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 2 --calc_option s_test --target None
	python3 influence_function_test.py --option fair --sen_attr sex --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 2 --calc_option influence --target None
	python3 influence_function_test.py --option val_loss --sen_attr sex --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 2 --calc_option s_test --target None
	python3 influence_function_test.py --option val_loss --sen_attr sex --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 2 --calc_option influence --target None

	for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
	do
		python3 main_final.py --dataset compas --method naive_leave_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 2 --k $k --sen_attr sex
		python3 main_final.py --dataset compas --method naive_leave_bottom_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 2 --k $k --sen_attr sex
	done
done


for seed in 0 1 2 3 4 5 6 7 8 9
do
	python3 main_final.py --sen_attr race --dataset compas --method naive --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 2
	python3 influence_function_test.py --option fair --sen_attr race --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 2 --calc_option grad_V --target None
	python3 influence_function_test.py --option fair --sen_attr race --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 2 --calc_option s_test --target None
	python3 influence_function_test.py --option fair --sen_attr race --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 2 --calc_option influence --target None
	python3 influence_function_test.py --option val_loss --sen_attr race --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 2 --calc_option s_test --target None
	python3 influence_function_test.py --option val_loss --sen_attr race --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 2 --calc_option influence --target None

	for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
	do
		python3 main_final.py --dataset compas --method naive_leave_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 2 --k $k --sen_attr race
		python3 main_final.py --dataset compas --method naive_leave_bottom_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 2 --k $k --sen_attr race
	done
done
