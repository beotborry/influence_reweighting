#!/bin/zsh

for seed in 100 200 300 777
do
  python main.py --method naive --dataset adult --epoch 20 --iteration 15 --constraint dp --seed $seed
  python main.py --method naive --dataset compas --epoch 20 --iteration 15 --constraint dp --seed $seed
done