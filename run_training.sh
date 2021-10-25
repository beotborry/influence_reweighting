#!/bin/zsh

python main.py --method reweighting --constraint eo --dataset compas --epoch 20 --iteration 20 --eta 1
python main.py --method reweighting --constraint eo --dataset compas --epoch 20 --iteration 20 --eta 2
python main.py --method reweighting --constraint eo --dataset compas --epoch 20 --iteration 20 --eta 3
python main.py --method reweighting --constraint eo --dataset compas --epoch 20 --iteration 20 --eta 4
python main.py --method reweighting --constraint eo --dataset compas --epoch 20 --iteration 20 --eta 5
python main.py --method reweighting --constraint eo --dataset compas --epoch 20 --iteration 20 --eta 6
python main.py --method reweighting --constraint eo --dataset compas --epoch 20 --iteration 20 --eta 7
python main.py --method reweighting --constraint eo --dataset compas --epoch 20 --iteration 20 --eta 8
python main.py --method reweighting --constraint eo --dataset compas --epoch 20 --iteration 20 --eta 9
python main.py --method reweighting --constraint eo --dataset compas --epoch 20 --iteration 20 --eta 10


python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 5
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 10
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 15
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 20

python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 25
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 30
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 35
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 40

python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 45
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 50
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 55
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 60


python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 65
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 70
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 75
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 80

python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 85
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 90
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 95
python main.py --method influence --constraint eo --dataset compas --epoch 20 --iteration 20 --scaler 100