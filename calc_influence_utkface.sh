#!/bin/zsh

for seed in 100 200 300 400
do
	for k in $(seq 10 10 200)
	do
		python3 main_image.py --gpu 1 --dataset utkface --constraint eopp --seed $seed --target None --epoch 25 --iteration 1 --method naive_leave_k_out --k $k
		python3 main_image.py --gpu 1 --dataset utkface --constraint eopp --seed $seed --target None --epoch 25 --iteration 1 --method naive_leave_bottom_k_out --k $k
	done

done


