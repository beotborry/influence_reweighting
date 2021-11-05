#!/bin/zsh

for eta in 1 2 3 4 5
do
  python main.py --method reweighting --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 100 --naive_acc 86.03 --naive_vio 16.47 --eta $eta
  python main.py --method reweighting --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 200 --naive_acc 86.08 --naive_vio 16.6 --eta $eta
  python main.py --method reweighting --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 300 --naive_acc 85.93 --naive_vio 17.05 --eta $eta
  python main.py --method reweighting --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 777 --naive_acc 85.98 --naive_vio 16.68 --eta $eta
  python main.py --method reweighting --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 100 --naive_acc 69.6 --naive_vio 22.84 --eta $eta
  python main.py --method reweighting --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 200 --naive_acc 69.65 --naive_vio 16.18 --eta $eta
  python main.py --method reweighting --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 300 --naive_acc 69.65 --naive_vio 24.36 --eta $eta
  python main.py --method reweighting --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 777 --naive_acc 69.32 --naive_vio 23.08 --eta $eta

done

for k in 50 100 150 200 250 300
do
  python main.py --method naive_leave_k_out --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 100 --naive_acc 86.03 --naive_vio 16.47 --k $k
  python main.py --method naive_leave_k_out --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 200 --naive_acc 86.08 --naive_vio 16.6 --k $k
  python main.py --method naive_leave_k_out --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 300 --naive_acc 85.93 --naive_vio 17.05 --k $k
  python main.py --method naive_leave_k_out --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 777 --naive_acc 85.98 --naive_vio 16.68 --k $k

  python main.py --method naive_leave_k_out --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 100 --naive_acc 69.6 --naive_vio 22.84 --k $k
  python main.py --method naive_leave_k_out --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 200 --naive_acc 69.65 --naive_vio 16.18 --k $k
  python main.py --method naive_leave_k_out --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 300 --naive_acc 69.65 --naive_vio 24.36 --k $k
  python main.py --method naive_leave_k_out --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 777 --naive_acc 69.32 --naive_vio 23.08 --k $k

done

for scaler in 5 10 15 20 25 30 35
do
  python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 100 --naive_acc 86.03 --naive_vio 16.47 --scaler $scaler --r 25 --t 1500
  python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 200 --naive_acc 86.08 --naive_vio 16.60 --scaler $scaler --r 25 --t 1500
  python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 300 --naive_acc 85.93 --naive_vio 17.05 --scaler $scaler --r 25 --t 1500
  python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 777 --naive_acc 85.98 --naive_vio 16.68 --scaler $scaler --r 25 --t 1500

  python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 100 --naive_acc 69.60 --naive_vio 22.84 --scaler $scaler --r 5 --t 1000
  python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 200 --naive_acc 69.65 --naive_vio 16.18 --scaler $scaler --r 5 --t 1000
  python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 300 --naive_acc 69.65 --naive_vio 24.36 --scaler $scaler --r 5 --t 1000
  python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 777 --naive_acc 69.32 --naive_vio 23.08 --scaler $scaler --r 5 --t 1000
done

for k in 50 100 150 200 250 300
do
  python main.py --method naive_leave_bottom_k_out --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 100 --naive_acc 86.03 --naive_vio 16.47 --k $k
  python main.py --method naive_leave_bottom_k_out --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 200 --naive_acc 86.08 --naive_vio 16.6 --k $k
  python main.py --method naive_leave_bottom_k_out --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 300 --naive_acc 85.93 --naive_vio 17.05 --k $k
  python main.py --method naive_leave_bottom_k_out --dataset adult --epoch 20 --iteration 15 --constraint dp --seed 777 --naive_acc 85.98 --naive_vio 16.68 --k $k

  python main.py --method naive_leave_bottom_k_out --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 100 --naive_acc 69.6 --naive_vio 22.84 --k $k
  python main.py --method naive_leave_bottom_k_out --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 200 --naive_acc 69.65 --naive_vio 16.18 --k $k
  python main.py --method naive_leave_bottom_k_out --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 300 --naive_acc 69.65 --naive_vio 24.36 --k $k
  python main.py --method naive_leave_bottom_k_out --dataset compas --epoch 20 --iteration 15 --constraint dp --seed 777 --naive_acc 69.32 --naive_vio 23.08 --k $k
done
