#!/bin/bash

python3 main_final.py --dataset adult --method naive_leave_k_out --seed 2 --constraint eopp --epoch 50 --iteration 1 --gpu 0 --k 0.5 --sen_attr sex --fine_tuning 0 --main_option intersect
python3 main_final.py --dataset adult --method naive_leave_bottom_k_out --seed 1 --constraint eopp --epoch 50 --iteration 1 --gpu 0 --k 0.5 --sen_attr sex --fine_tuning 0 --main_option intersect
python3 main_final.py --dataset compas --method naive_leave_k_out --seed 3 --constraint eopp --epoch 50 --iteration 1 --gpu 0 --k 7.0 --sen_attr sex --fine_tuning 0 --main_option intersect
python3 main_final.py --dataset adult --method naive_leave_bottom_k_out --seed 1 --constraint eopp --epoch 50 --iteration 1 --gpu 0 --k 7.0 --sen_attr sex --fine_tuning 0 --main_option intersect
python3 main_final.py --dataset bank --method naive_leave_k_out --seed 2 --constraint eopp --epoch 50 --iteration 1 --gpu 0 --k 1.0 --sen_attr age --fine_tuning 0 --main_option intersect
