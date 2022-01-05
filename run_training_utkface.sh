#!/bin/zsh

python3 main_image.py --gpu 2 --dataset utkface --method naive --constraint eopp --epoch 25 --iteration 1 --seed 100

for k in 50 100 150 200 250 300 350 400 450 500
do
	python3 main_image.py --gpu 2 --dataset utkface --method naive_leave_bottom_k_out --k $k --constraint eopp --epoch 25 --iteration 1 --seed 100
	python3 main_image.py --gpu 2 --dataset utkface --method naive_leave_bottom_k_out --k $k --constraint eopp --epoch 25 --iteration 1 --seed 100
done

