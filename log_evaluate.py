import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
from model_evaluate import model_evaluate

def get_args():
    parser = argparse.ArgumentParser(description="log evaluation")
    parser.add_argument('--option', required=True, default='', choices=['top', 'bottom', 'random'])
    parser.add_argument('--dataset', required=True, default='', choices=['utkface', 'celeba'])
    parser.add_argument('--target', required=True)
    parser.add_argument('--constraint', required=True, choices=['eopp', 'eo', 'dp'])
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--seed', required=True, default=None, type=int)

    args = parser.parse_args()
    return args

args = get_args()
option = args.option
dataset = args.dataset
seed = args.seed
target = args.target
constraint = args.constraint
gpu = args.gpu

naive_acc, naive_fair = model_evaluate(dataset, target, constraint, seed, gpu)
print("Naive Acc: {:.2f}, Naive Fair: {:.2f}".format(naive_acc, naive_fair))


for k in range(10, 210, 10):
    if option == "top":
        filename = "./image_log/{}_seed_{}_k_{}_log.txt"
    elif option == "bottom":
        filename = "./image_log/{}_seed_{}_bottom_k_{}_log.txt"

    with open(filename.format(dataset, seed, k), "rb") as fp:
        log = pickle.load(fp)
        best_tradeoff = 0.0
        best_idx = 0

        best_acc = 0.0
        i = 0

        candidate_idx = []

        for acc, fair in zip(log[2], log[3]):
            # # print("k: {}, test_acc: {:.2f}, fair: {:.2f}".format(k, acc * 100, fair * 100))
            # if acc * 100 >= naive_acc:
            #     #print("k: {}, acc: {:.2f}, fair: {:.2f}".format(k, acc * 100, fair * 100))
            #     candidate_idx.append(i)
            # else:
            #     # if 'bottom' not in filename:
            #     #     cur_tradeoff = (naive_fair - fair * 100) / (naive_acc - acc * 100)
            #     #     if cur_tradeoff >= best_tradeoff:
            #     #         best_tradeoff = cur_tradeoff
            #     #         best_idx = i
            #     # else:
            #     #     if acc * 100 >= best_acc:
            #     #         beat_acc = acc * 100
            #     #         best_idx = i

            if acc * 100 >= best_acc:
                best_acc = acc * 100
                best_idx = i

            i += 1

        
        
        # if 'bottom' not in filename:
        #     for idx in candidate_idx:
        #         candidate_acc = log[2][idx]
        #         candidate_fair = log[3][idx]

        #         if candidate_acc > log[2][best_idx] and candidate_fair < log[3][best_idx]:
        #             best_idx = idx

        # if 'bottom' not in filename:
        #     print("k:{}, best tradeoff acc:{:.2f}, fair: {:.2f}".format(k, log[2][best_idx] * 100, log[3][best_idx] * 100))
        # else:
        #     print("k:{}, best acc: {:.2f}, fair: {:.2f}".format(k, log[2][best_idx] * 100, log[3][best_idx] * 100))
        print("k:{}, best acc: {:.2f}, fair: {:.2f}".format(k, log[2][best_idx] * 100, log[3][best_idx] * 100))

        #print("k:{}, best acc: {:.2f}, fair: {:.2f}".format(k, log[2][best_idx] * 100, log[3][best_idx] * 100))

        
    
