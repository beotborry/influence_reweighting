#!/bin/zsh

for k in $(seq 250 50 300)
do
    python3 main.py --gpu 1 --dataset adult --constraint eopp --method naive --idx_save 1 --seed 1 --k $k --epoch 20 --iteration 1 --r 27 --t 1000
done

