#!/bin/zsh


python3 main_image.py --dataset utkface --method naive --epoch 20 --iteration 1 --constraint eopp --seed 100

python3 model_evaluate.py --dataset utkface --method naive --epoch 20 --iteration 1 --constraint eopp --seed 100
