#!/bin/zsh

python3 main_image.py --dataset utkface --epoch 20 --seed 100 --method naive --target
python3 model_evaluate.py --dataset utkface --constraint eopp --seed 100 --target 


