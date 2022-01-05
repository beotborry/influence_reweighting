#!/bin/zsh

python3 influence_function_test.py --dataset celeba --constraint eopp --seed 100 --target Attractive --calc_option grad_V --r 450 --t 400 --gpu 3
python3 influence_function_test.py --dataset celeba --constraint eopp --seed 100 --target Attractive --calc_option s_test --r 450 --t 400 --gpu 3
python3 influence_function_test.py --dataset celeba --constraint eopp --seed 100 --target Attractive --calc_option influence --r 450 --t 400 --gpu 3


