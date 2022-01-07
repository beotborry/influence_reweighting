#!/bin/zsh

for seed in 400
do

	python3 main_image.py --dataset celeba --constraint eopp --seed $seed --target Attractive --epoch 25 --iteration 1 --method naive --gpu 2

	python3 influence_function_test.py --dataset celeba --constraint eopp --seed $seed  --target Attractive --calc_option grad_V --r 450 --t 400 --gpu 2
	python3 influence_function_test.py --dataset celeba --constraint eopp --seed $seed --target Attractive --calc_option s_test --r 450 --t 400 --gpu 2
	python3 influence_function_test.py --dataset celeba --constraint eopp --seed $seed --target Attractive --calc_option influence --r 450 --t 400 --gpu 2
		
	for k in $(seq 10 10 250)
	do
		python3 main_image.py --dataset celeba --constraint eopp --seed $seed --target Attractive --epoch 25 --iteration 1 --method naive_leave_k_out --gpu 0 --k $k
		python3 main_image.py --dataset celeba --constraint eopp --seed $seed --target Attractive --epoch 25 --iteration 1 --method naive_leave_bottom_k_out --gpu 0 --k $k
	done
done




