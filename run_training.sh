#!/bin/zsh

python3 main_image.py --dataset celeba --constraint eopp --seed 100 --target Young --method naive_leave_k_out --k 50 --epoch 20 --iteration 1

python3 main_image.py --dataset celeba --constraint eopp --seed 100 --target Young --method naive_leave_k_out --k 100 --epoch 20 --iteration 1

python3 main_image.py --dataset celeba --constraint eopp --seed 100 --target Young --method naive_leave_k_out --k 150 --epoch 20 --iteration 1
python3 main_image.py --dataset celeba --constraint eopp --seed 100 --target Young --method naive_leave_k_out --k 200 --epoch 20 --iteration 1
python3 main_image.py --dataset celeba --constraint eopp --seed 100 --target Young --method naive_leave_k_out --k 250 --epoch 20 --iteration 1

python3 main_image.py --dataset celeba --constraint eopp --seed 100 --target Young --method naive_leave_k_out --k 300 --epoch 20 --iteration 1

