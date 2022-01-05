#!/bin/zsh

python3 influence_function_test.py --gpu 2 --dataset utkface --constraint eopp --seed 100 --target None --calc_option grad_V --r 30 --t 400
python3 influence_function_test.py --gpu 2 --dataset utkface --constraint eopp --seed 100 --target None --calc_option s_test --r 30 --t 400
python3 influence_function_test.py --gpu 2 --dataset utkface --constraint eopp --seed 100 --target None --calc_option influence --r 30 --t 400

