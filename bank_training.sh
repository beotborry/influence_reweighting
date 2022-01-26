#!/bin/bash

for seed in 0 1 2 3 4
do
	python3 main_final.py --dataset bank --method naive --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 1 --sen_attr age --fine_tuning 0 --main_option fair_only
	python3 calc_influence.py --option fair --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option grad_V --target None --sen_attr age --main_option fair_only
	python3 calc_influence.py --option fair --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option s_test --target None --sen_attr age --main_option fair_only
	python3 calc_influence.py --option fair --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option influence --target None --sen_attr age --main_option fair_only
	
	for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
	do
		python3 main_final.py --dataset bank --method naive_leave_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 1 --k $k --sen_attr age --fine_tuning 0 --main_option fair_only
		python3 main_final.py --dataset bank --method naive_leave_bottom_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 1 --k $k --sen_attr age --fine_tuning 0 --main_option fair_only
	done
done


for seed in 0 1 2 3 4
do
	python3 main_final.py --dataset bank --method naive --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 1 --sen_attr age --fine_tuning 0 --main_option fair_only_fine_tuning
	python3 calc_influence.py --option fair --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option grad_V --target None --sen_attr age --main_option fair_only_fine_tuning
	python3 calc_influence.py --option fair --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option s_test --target None --sen_attr age --main_option fair_only_fine_tuning
	python3 calc_influence.py --option fair --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option influence --target None --sen_attr age --main_option fair_only_fine_tuning
	
	for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
	do
		python3 main_final.py --dataset bank --method naive_leave_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 1 --k $k --sen_attr age --fine_tuning 1 --main_option fair_only_fine_tuning
		python3 main_final.py --dataset bank --method naive_leave_bottom_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 1 --k $k --sen_attr age --fine_tuning 1 --main_option fair_only_fine_tuning
	done
done

for seed in 0 1 2 3 4
do
	python3 main_final.py --dataset bank --method naive --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 1 --sen_attr age --fine_tuning 0 --main_option intersect
	python3 calc_influence.py --option fair --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option grad_V --target None --sen_attr age --main_option intersect
	python3 calc_influence.py --option fair --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option s_test --target None --sen_attr age --main_option intersect
	python3 calc_influence.py --option fair --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option influence --target None --sen_attr age --main_option intersect
	python3 calc_influence.py --option val_loss --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option s_test --target None --sen_attr age --main_option intersect
	python3 calc_influence.py --option val_loss --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option influence --target None --sen_attr age --main_option intersect
	
	for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
	do
		python3 main_final.py --dataset bank --method naive_leave_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 1 --k $k --sen_attr age --fine_tuning 0 --main_option intersect
		python3 main_final.py --dataset bank --method naive_leave_bottom_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 1 --k $k --sen_attr age --fine_tuning 0 --main_option intersect
	done
done
for seed in 0 1 2 3 4
do
	python3 main_final.py --dataset bank --method naive --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 1 --sen_attr age --fine_tuning 0 --main_option intersect_fine_tuning
	python3 calc_influence.py --option fair --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option grad_V --target None --sen_attr age --main_option intersect_fine_tuning
	python3 calc_influence.py --option fair --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option s_test --target None --sen_attr age --main_option intersect_fine_tuning
	python3 calc_influence.py --option fair --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option influence --target None --sen_attr age --main_option intersect_fine_tuning
	python3 calc_influence.py --option val_loss --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option s_test --target None --sen_attr age --main_option intersect_fine_tuning
	python3 calc_influence.py --option val_loss --dataset bank --seed $seed --constraint eopp --r 20 --t 1000 --gpu 1 --calc_option influence --target None --sen_attr age --main_option intersect_fine_tuning
	
	for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
	do
		python3 main_final.py --dataset bank --method naive_leave_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 1 --k $k --sen_attr age --fine_tuning 1 --main_option intersect_fine_tuning
		python3 main_final.py --dataset bank --method naive_leave_bottom_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 1 --k $k --sen_attr age --fine_tuning 1 --main_option intersect_fine_tuning
	done
done
