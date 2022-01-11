import pickle
import matplotlib.pyplot as plt
import argparse
from utils import set_seed

def get_args():
    parser = argparse.ArgumentParser(description="log evaluation")
    parser.add_argument('--option', required=True, default='', choices=['top', 'bottom', 'random'])
    parser.add_argument('--dataset', required=True, default='', choices=['adult', 'utkface', 'celeba'])
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

set_seed(seed)

if option == "top":
    filename = "./image_log/{}_seed_{}_k_{}_log.txt"
elif option == "bottom":
    filename = "./image_log/{}_seed_{}_bottom_k_{}_log.txt"

for k in range(10, 500, 10):
    try:
        print("here")
        with open(filename.format(dataset, seed, k), "rb") as fp:
            log = pickle.load(fp)
            print(log[4], log[5])
            plt.scatter(log[4], log[5], c=k)
    except:
        continue
    
plt.show()
plt.savefig('./log.png')
