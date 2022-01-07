#!/bin/zsh

for k in $(seq 10 10 250)
do
	python3 main_image.py --gpu 2 --dataset utkface --method naive_leave_k_out --k $k --constraint eopp --epoch 25 --iteration 1 --seed 100
	python3 main_image.py --gpu 2 --dataset utkface --method naive_leave_bottom_k_out --k $k --constraint eopp --epoch 25 --iteration 1 --seed 100
done

