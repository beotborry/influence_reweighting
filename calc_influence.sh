#!/bin/zsh

for seed in 300
do		
	for k in $(seq 50 50 500)
	do
		python3 main_image.py --dataset celeba --constraint eopp --seed $seed --target Attractive --epoch 25 --iteration 1 --method naive_leave_k_out --gpu 2 --k $k
		python3 main_image.py --dataset celeba --constraint eopp --seed $seed --target Attractive --epoch 25 --iteration 1 --method naive_leave_bottom_k_out --gpu 2 --k $k
	done
done




