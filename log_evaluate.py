import pickle
import numpy as np

naive_acc = 75.75
naive_fair = 6.5

for k in range(50, 950, 50):
    with open("./image_log/utkface_seed_100_bottom_k_{}_log.txt".format(k), "rb") as fp:
        log = pickle.load(fp)
        best_tradeoff = 0.0
        best_idx = 0
        i = 0

        for acc, fair in zip(log[0], log[1]):
            if acc * 100 >= naive_acc:
                print("k: {}, acc: {:.2f}, fair: {:.2f}".format(k, acc * 100, fair * 100))
            else:
                if (naive_fair - fair * 100) / (naive_acc -acc * 100) >= best_tradeoff:
                    best_tradeoff = (naive_fair - fair * 100) / (naive_acc - acc * 100)
                    best_idx = i
            i += 1

        print("k:{}, best tradeoff acc:{:.2f}, fair: {:.2f}".format(k, log[0][best_idx] * 100, log[1][best_idx] * 100))

    
