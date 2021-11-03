#!/bin/zsh

python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eopp --scaler 1
python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eopp --scaler 2
python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eopp --scaler 3
python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eopp --scaler 4
