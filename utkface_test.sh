#!/bin/bash

for seed in 0 1 2 3 4
do
	# python3 main_final.py --dataset utkface --method naive --seed $seed --constraint eopp --epoch 1 --iteration 1 --gpu 0 --sen_attr race --fine_tuning 0 --main_option fair_only --log_option all --target age
	# python3 calc_influence.py --option fair --dataset utkface --seed $seed --constraint eopp --r 10 --t 1000 --gpu 0 --calc_option grad_V --target age --sen_attr race  --main_option fair_only
	# python3 calc_influence.py --option fair --dataset utkface --seed $seed --constraint eopp --r 10 --t 1000 --gpu 0 --calc_option s_test --target age --sen_attr race --main_option fair_only
	# python3 calc_influence.py --option fair --dataset utkface --seed $seed --constraint eopp --r 10 --t 1000 --gpu 0 --calc_option influence --target age --sen_attr race --main_option fair_only
	# python3 calc_influence.py --option fair --dataset utkface --seed $seed --constraint eo --r 10 --t 1000 --gpu 0 --calc_option grad_V --target age --sen_attr race  --main_option fair_only
	# python3 calc_influence.py --option fair --dataset utkface --seed $seed --constraint eo --r 10 --t 1000 --gpu 0 --calc_option s_test --target age --sen_attr race --main_option fair_only
	# python3 calc_influence.py --option fair --dataset utkface --seed $seed --constraint eo --r 10 --t 1000 --gpu 0 --calc_option influence --target age --sen_attr race --main_option fair_only

	# python3 main_final.py --dataset utkface --method naive --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 0 --sen_attr race --fine_tuning 0 --main_option fair_only_split --log_option all --target age --batch_size 128
	python3 calc_influence.py --option fair --dataset utkface --seed $seed --constraint eopp --r 10 --t 1000 --gpu 0 --calc_option grad_V --target age --sen_attr race --main_option fair_only_split --batch_size 128
	python3 calc_influence.py --option fair --dataset utkface --seed $seed --constraint eopp --r 10 --t 1000 --gpu 0 --calc_option s_test --target age --sen_attr race --main_option fair_only_split --batch_size 128
	python3 calc_influence.py --option fair --dataset utkface --seed $seed --constraint eopp --r 10 --t 1000 --gpu 0 --calc_option influence --target age --sen_attr race --main_option fair_only_split --batch_size 128
	# python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eo --r 93 --t 10000 --gpu 2 --calc_option grad_V --target None --sen_attr race --main_option fair_only_split
	# python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eo --r 93 --t 10000 --gpu 2 --calc_option s_test --target None --sen_attr race --main_option fair_only_split
	# python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eo --r 93 --t 10000 --gpu 2 --calc_option influence --target None --sen_attr race --main_option fair_only_split
done
