#!/bin/bash

for seed in 100
do
	for k in $(seq 300 50 500)
	do 
		python3 main_image.py --gpu 2 --dataset utkface --constraint eopp --epoch 25 --iteration 1 --method naive_leave_k_out --k $k --seed $seed
		python3 main_image.py --gpu 2 --dataset utkface --constraint eopp --epoch 25 --iteration 1 --method naive_leave_bottom_k_out --k $k --seed $seed
	done
done

for seed in 200 300 400
do
	for k in $(seq 50 50 500)
	do 
		python3 main_image.py --gpu 2 --dataset utkface --constraint eopp --epoch 25 --iteration 1 --method naive_leave_k_out --k $k --seed $seed
		python3 main_image.py --gpu 2 --dataset utkface --constraint eopp --epoch 25 --iteration 1 --method naive_leave_bottom_k_out --k $k --seed $seed
	done
done

