#!/bin/zsh

python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint eopp --seed 100 --naive_acc 100 --naive_vio 100 --scaler 50 --r 25 --t 1500

for k in 50 100 150 200 250 300
do
	python main.py --method leave_random_k_out --dataset adult --epoch 20 --iteration 15 --constraint eopp --seed 100 --k $k
done

python main.py --method naive_leave_bottom_k_out --dataset compas --epoch 20 --iteration 15 --constraint eopp --seed 100 --k 200


python main.py --method naive_leave_k_out --dataset compas --epoch 20 --iteration 15 --constraint eopp --seed 100 --naive_acc 100 --naive_vio 100 --k 200
python main.py --method naive_leave_bottom_k_out --dataset compas --epoch 20 --iteration 15 --constraint eopp --seed 100 --k 200


for k in 500 510 520 530 540 550 560 570 580 590 600
do
	for i in 250,69.56,23.86
	do
		IFS=',' read seed acc vio <<< "${i}"
		python main.py --method naive --dataset compas --epoch 20 --iteration 15 --constraint dp --seed $seed --k $k --idx_save 1 --r 5 --t 1000
		python main.py --method naive_leave_k_out --dataset compas --epoch 20 --iteration 15 --constraint dp --seed $seed --naive_acc $acc --naive_vio $vio --k $k
		python main.py --method naive_leave_bottom_k_out --dataset compas --epoch 20 --iteration 15 --constraint dp --seed $seed --k $k
	done
done
