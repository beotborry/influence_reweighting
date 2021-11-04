#!/bin/zsh

for scaler in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 25 --t 1500 --naive_acc 86.03 --naive_vio 5.63 --seed 100
    python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 25 --t 1500 --naive_acc 86.08 --naive_vio 7.17 --seed 200
    python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 25 --t 1500 --naive_acc 85.93 --naive_vio 8.54 --seed 300
    python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 25 --t 1500 --naive_acc 85.98 --naive_vio 6.57 --seed 777
    python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 5 --t 1000 --naive_acc 69.6 --naive_vio 26.94 --seed 100
    python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 5 --t 1000 --naive_acc 69.65 --naive_vio 30.22 --seed 200
    python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 5 --t 1000 --naive_acc 69.65 --naive_vio 26.2 --seed 300
    python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 5 --t 1000 --naive_acc 69.32 --naive_vio 26.81 --seed 777
done

for eta in 1 2 3 4 5
do
    python main.py --method reweighting --dataset adult --epoch 20 --iteration 15 --constraint eo --eta $eta --naive_acc 86.03 --naive_vio 3.19 --seed 100
    python main.py --method reweighting --dataset adult --epoch 20 --iteration 15 --constraint eo --eta $eta --naive_acc 86.08 --naive_vio 3.98 --seed 200
    python main.py --method reweighting --dataset adult --epoch 20 --iteration 15 --constraint eo --eta $eta --naive_acc 85.93 --naive_vio 4.64 --seed 300
    python main.py --method reweighting --dataset adult --epoch 20 --iteration 15 --constraint eo --eta $eta --naive_acc 85.98 --naive_vio 3.68 --seed 777
    python main.py --method reweighting --dataset compas --epoch 20 --iteration 15 --constraint eo --eta $eta --naive_acc 69.6 --naive_vio 17.57 --seed 100
    python main.py --method reweighting --dataset compas --epoch 20 --iteration 15 --constraint eo --eta $eta --naive_acc 69.65 --naive_vio 18.33 --seed 200
    python main.py --method reweighting --dataset compas --epoch 20 --iteration 15 --constraint eo --eta $eta --naive_acc 69.65 --naive_vio 16.69 --seed 300
    python main.py --method reweighting --dataset compas --epoch 20 --iteration 15 --constraint eo --eta $eta --naive_acc 69.32 --naive_vio 17.55 --seed 777
done

for scaler in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 25 --t 1500 --naive_acc 86.03 --naive_vio 3.19 --seed 100
    python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 25 --t 1500 --naive_acc 86.08 --naive_vio 3.98 --seed 200
    python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 25 --t 1500 --naive_acc 85.93 --naive_vio 4.64 --seed 300
    python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 25 --t 1500 --naive_acc 85.98 --naive_vio 3.68 --seed 777
    python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 5 --t 1000 --naive_acc 69.6 --naive_vio 17.57 --seed 100
    python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 5 --t 1000 --naive_acc 69.65 --naive_vio 18.33 --seed 200
    python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 5 --t 1000 --naive_acc 69.65 --naive_vio 16.69 --seed 300
    python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eo --scaler $scaler --r 5 --t 1000 --naive_acc 69.32 --naive_vio 17.55 --seed 777
done
=======
for scaler in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint eopp --scaler $scaler --r 25 --t 1500 --naive_acc 86.03 --naive_vio 5.63 --seed 100
    python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint eopp --scaler $scaler --r 25 --t 1500 --naive_acc 86.08 --naive_vio 7.17 --seed 200
    python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint eopp --scaler $scaler --r 25 --t 1500 --naive_acc 85.93 --naive_vio 8.54 --seed 300
    python main.py --method influence --dataset adult --epoch 20 --iteration 15 --constraint eopp --scaler $scaler --r 25 --t 1500 --naive_acc 85.98 --naive_vio 6.57 --seed 777
    python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eopp --scaler $scaler --r 5 --t 1000 --naive_acc 69.6 --naive_vio 26.81 --seed 100
    python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eopp --scaler $scaler --r 5 --t 1000 --naive_acc 69.65 --naive_vio 30.22 --seed 200
    python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eopp --scaler $scaler --r 5 --t 1000 --naive_acc 69.65 --naive_vio 26.2 --seed 300
    python main.py --method influence --dataset compas --epoch 20 --iteration 15 --constraint eopp --scaler $scaler --r 5 --t 1000 --naive_acc 69.32 --naive_vio 26.81 --seed 777
done
>>>>>>> Stashed changes
