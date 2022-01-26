#!/bin/bash

for seed in 0
do 
	python3 main_final.py --dataset retiring_adult --method naive --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 3 --sen_attr sex --fine_tuning 0
	python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eopp --r 107 --t 10000 --gpu 3 --calc_option grad_V --target None --sen_attr sex
	python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eopp --r 107 --t 10000 --gpu 3 --calc_option s_test --target None --sen_attr sex
	python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eopp --r 107 --t 10000 --gpu 3 --calc_option influence --target None --sen_attr sex
	python3 calc_influence.py --option val_loss --dataset retiring_adult --seed $seed --constraint eopp --r 107 --t 10000 --gpu 3 --calc_option s_test --target None --sen_attr sex
	python3 calc_influence.py --option val_loss --dataset retiring_adult --seed $seed --constraint eopp --r 107 --t 10000 --gpu 3 --calc_option influence --target None --sen_attr sex


	for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
	do
		python3 main_final.py --dataset retiring_adult --method naive_leave_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 3 --k $k --sen_attr sex --fine_tuning 1
		python3 main_final.py --dataset retiring_adult --method naive_leave_bottom_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 3 --k $k --sen_attr sex --fine_tuning 1
	done
done

for seed in 1
do 
	python3 main_final.py --dataset retiring_adult --method naive --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 3 --sen_attr sex --fine_tuning 0
	python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eopp --r 107 --t 10000 --gpu 3 --calc_option grad_V --target None --sen_attr sex
	python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eopp --r 107 --t 10000 --gpu 3 --calc_option s_test --target None --sen_attr sex
	python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eopp --r 107 --t 10000 --gpu 3 --calc_option influence --target None --sen_attr sex
	python3 calc_influence.py --option val_loss --dataset retiring_adult --seed $seed --constraint eopp --r 107 --t 10000 --gpu 3 --calc_option s_test --target None --sen_attr sex
	python3 calc_influence.py --option val_loss --dataset retiring_adult --seed $seed --constraint eopp --r 107 --t 10000 --gpu 3 --calc_option influence --target None --sen_attr sex


	for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
	do
		python3 main_final.py --dataset retiring_adult --method naive_leave_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 3 --k $k --sen_attr sex --fine_tuning 1
		python3 main_final.py --dataset retiring_adult --method naive_leave_bottom_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 3 --k $k --sen_attr sex --fine_tuning 1
	done
done
