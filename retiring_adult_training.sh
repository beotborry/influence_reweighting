#!/bin/bash

for seed in 123
do 
	# python3 main_final.py --dataset retiring_adult --method naive --seed $seed --constraint eopp --epoch 30 --iteration 1 --gpu 1 --sen_attr race --fine_tuning 0 --main_option fair_only_split --log_option all --batch_size 2048
	python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eopp --r 93 --t 10000 --gpu 1 --calc_option grad_V --target None --sen_attr race --main_option fair_only_split --batch_size 2048
	python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eopp --r 93 --t 10000 --gpu 1 --calc_option s_test --target None --sen_attr race --main_option fair_only_split --batch_size 2048
	python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eopp --r 93 --t 10000 --gpu 1 --calc_option influence --target None --sen_attr race --main_option fair_only_split --batch_size 2048
	# python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eo --r 93 --t 10000 --gpu 2 --calc_option grad_V --target None --sen_attr race --main_option fair_only_split --batch_size 2048
	# python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eo --r 93 --t 10000 --gpu 2 --calc_option s_test --target None --sen_attr race --main_option fair_only_split --batch_size 2048
	# python3 calc_influence.py --option fair --dataset retiring_adult --seed $seed --constraint eo --r 93 --t 10000 --gpu 2 --calc_option influence --target None --sen_attr race --main_option fair_only_split --batch_size 2048

	# cp "./model/fair_only/retiring_adult_MLP_target_None_seed_${seed}_sen_attr_race" "./model/fair_only_fine_tuning"
	# cp "./model/fair_only/retiring_adult_MLP_target_None_seed_${seed}_sen_attr_race" "./model/intersect"
	# cp "./model/fair_only/retiring_adult_MLP_target_None_seed_${seed}_sen_attr_race" "./model/intersect_fine_tuning"



	# cp "./influence_score/fair_only/retiring_adult_influence_score_seed_${seed}_sen_attr_race.txt" "./influence_score/fair_only_fine_tuning"
	# cp "./influence_score/fair_only/retiring_adult_influence_score_seed_${seed}_sen_attr_race.txt" "./influence_score/intersect"
	# cp "./influence_score/fair_only/retiring_adult_influence_score_seed_${seed}_sen_attr_race.txt" "./influence_score/intersect_fine_tuning"
	# cp "./influence_score/fair_only/retiring_adult_val_loss_influence_score_seed_${seed}_sen_attr_race.txt" "./influence_score/intersect"
	# cp "./influence_score/fair_only/retiring_adult_val_loss_influence_score_seed_${seed}_sen_attr_race.txt" "./influence_score/intersect_fine_tuning"


	# for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
	# do
	# 	for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
	# 	do
		
	# 		python3 main_final.py --dataset retiring_adult --method naive_leave_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 2 --k $k --sen_attr race --fine_tuning 0 --main_option fair_with_val_loss --log_option last --alpha $alpha
	# 		# python3 main_final.py --dataset retiring_adult --method naive_leave_bottom_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 2 --k $k --sen_attr race --fine_tuning 0 --main_option fair_with_val_loss --log_option last --alpha $alpha
	# 		python3 main_final.py --dataset retiring_adult --method naive_leave_k_out --seed $seed --constraint eo --epoch 50 --iteration 1 --gpu 2 --k $k --sen_attr race --fine_tuning 0 --main_option fair_with_val_loss --log_option last --alpha $alpha
	# 		# python3 main_final.py --dataset retiring_adult --method naive_leave_bottom_k_out --seed $seed --constraint eo --epoch 50 --iteration 1 --gpu 2 --k $k --sen_attr race --fine_tuning 0 --main_option fair_with_val_loss --log_option last --alpha $alpha

	# 	done
	# done
done



