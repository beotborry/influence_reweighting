#!/bin/zsh

for seed in 200 300 400
do
	python3 influence_function_test.py --gpu 1 --dataset utkface --constraint eopp --seed $seed --target None --calc_option influence

	for k in $(seq 10 10 200)
	do 
		python3 main_image.py --gpu 1 --dataset utkface --constraint eopp --epoch 25 --iteration 1 --method naive_leave_k_out --k $k --seed $seed
		python3 main_image.py --gpu 1 --dataset utkface --constraint eopp --epoch 25 --iteration 1 --method naive_leave_bottom_k_out --k $k --seed $seed

	done
done

