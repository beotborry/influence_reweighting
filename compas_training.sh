#!/bin/bash

for seed in 0 1 2 3 4
do

	# python3 main_final.py --sen_attr sex --dataset compas --method naive --seed $seed --constraint eo --epoch 50 --iteration 1 --gpu 1 --fine_tuning 0 --main_option fair_only --log_option all
	# python3 calc_influence.py --option fair --sen_attr sex --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 1 --calc_option grad_V --target None --main_option fair_only
	# python3 calc_influence.py --option fair --sen_attr sex --dataset compas --seed $seed --constraint eo --r 5 --t 1000 --gpu 1 --calc_option grad_V --target None --main_option fair_only
	# python3 calc_influence.py --option fair --sen_attr sex --dataset compas --seed $seed --constraint eo --r 5 --t 1000 --gpu 1 --calc_option s_test --target None --main_option fair_only
	# python3 calc_influence.py --option fair --sen_attr sex --dataset compas --seed $seed --constraint eo --r 5 --t 1000 --gpu 1 --calc_option influence --target None --main_option fair_only
	# python3 calc_influence.py --option val_loss --sen_attr sex --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 1 --calc_option s_test --target None --main_option fair_only
	# python3 calc_influence.py --option val_loss --sen_attr sex --dataset compas --seed $seed --constraint eopp --r 5 --t 1000 --gpu 1 --calc_option influence --target None --main_option fair_only


	# cp "./model/fair_only/compas_MLP_target_None_seed_${seed}_sen_attr_sex" "./model/fair_only_fine_tuning"
	# cp "./model/fair_only/compas_MLP_target_None_seed_${seed}_sen_attr_sex" "./model/intersect"
	# cp "./model/fair_only/compas_MLP_target_None_seed_${seed}_sen_attr_sex" "./model/intersect_fine_tuning"



	# cp "./influence_score/fair_only/compas_influence_score_seed_${seed}_sen_attr_sex.txt" "./influence_score/fair_only_fine_tuning"
	# cp "./influence_score/fair_only/compas_influence_score_seed_${seed}_sen_attr_sex.txt" "./influence_score/intersect"
	# cp "./influence_score/fair_only/compas_influence_score_seed_${seed}_sen_attr_sex.txt" "./influence_score/intersect_fine_tuning"
	# cp "./influence_score/fair_only/compas_val_loss_influence_score_seed_${seed}_sen_attr_sex.txt" "./influence_score/intersect"
	# cp "./influence_score/fair_only/compas_val_loss_influence_score_seed_${seed}_sen_attr_sex.txt" "./influence_score/intersect_fine_tuning"

	for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
	do
		for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
		do
			python3 main_final.py --dataset compas --method naive_leave_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 1 --k $k --sen_attr sex --fine_tuning 0 --main_option fair_with_val_loss --log_option last --alpha $alpha
			python3 main_final.py --dataset compas --method naive_leave_bottom_k_out --seed $seed --constraint eopp --epoch 50 --iteration 1 --gpu 1 --k $k --sen_attr sex --fine_tuning 0 --main_option fair_with_val_loss --log_option last --alpha $alpha
			python3 main_final.py --dataset compas --method naive_leave_k_out --seed $seed --constraint eo --epoch 50 --iteration 1 --gpu 1 --k $k --sen_attr sex --fine_tuning 0 --main_option fair_with_val_loss --log_option last --alpha $alpha
			python3 main_final.py --dataset compas --method naive_leave_bottom_k_out --seed $seed --constraint eo --epoch 50 --iteration 1 --gpu 1 --k $k --sen_attr sex --fine_tuning 0 --main_option fair_with_val_loss --log_option last --alpha $alpha

		done
	done
done


