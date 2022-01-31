#!/bin/bash

for seed in 0 1
do 

	for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
	do
		python3 main_final.py --dataset retiring_adult --method naive_leave_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 0 --k $k --sen_attr race --fine_tuning 1 --main_option intersect_fine_tuning
		python3 main_final.py --dataset retiring_adult --method naive_leave_bottom_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 0 --k $k --sen_attr race --fine_tuning 1 --main_option intersect_fine_tuning
	done
done



