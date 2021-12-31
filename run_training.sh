#!/bin/zsh

python3 main_image.py --dataset utkface --epoch 20 --seed 100 --method naive

python3 main_image.py --dataset celeba --epoch 20 --iteration 15 --constraint eopp --seed 100 --method naive


