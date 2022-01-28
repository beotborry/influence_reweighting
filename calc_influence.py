import torch
from data_handler.dataloader_factory import DataloaderFactory
from influence_function_image import avg_s_test, grad_V, calc_influence_dataset
import pickle
from utils import set_seed
from torch.utils.data import DataLoader
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="get influence scores for images")
    parser.add_argument('--gpu', required=True, default=3, type=int)
    parser.add_argument('--dataset', required=True, default='', choices=['adult', 'compas', 'bank', 'celeba', 'utkface', 'retiring_adult', 'retiring_adult_coverage'])
    parser.add_argument('--sen_attr', default='sex')
    parser.add_argument('--target', required=True, default='')
    parser.add_argument('--constraint', required=True, default='', choices=['dp', 'eo', 'eopp'])
    parser.add_argument('--calc_option', required=True, default='', choices=['grad_V', 's_test', 'influence'])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--option', required=True)
    parser.add_argument('--main_option', required=True, choices=['fair_only', 'fair_only_fine_tuning', 'intersect', 'intersect_fine_tuning'])
    parser.add_argument('--r', default = None, type=int)
    parser.add_argument('--t', default = None, type=int)

    args = parser.parse_args()

    return args

def get_grad_V(constraint, dataloader, model, _dataset, _seed, _sen_attr, main_option, save=True):
    grad_V(constraint, dataloader, model, _dataset, _seed, _sen_attr, main_option, save=save)

def get_avg_s_test(model, dataloader, random_sampler, constraint, weights, r, _dataset, _seed, _sen_attr, recursion_depth, option, main_option, save=True):
    avg_s_test(model=model, dataloader=dataloader, random_sampler=random_sampler, constraint=constraint, weights=weights, r=r, recursion_depth=recursion_depth, _dataset=_dataset, _seed=_seed, _sen_attr=_sen_attr, save=save, option=option, main_option=main_option)

def get_influence_score(model, dataloader, s_test_dataloader, random_sampler, constraint, weights, _dataset, _seed, _sen_attr, recursion_depth, r, option, main_option, load_s_test=True):
    influences = calc_influence_dataset(model=model, dataloader=dataloader, s_test_dataloader = s_test_dataloader, random_sampler=random_sampler, constraint=constraint, weights=weights, recursion_depth=recursion_depth, r=r, _dataset=_dataset, _seed=_seed, _sen_attr=_sen_attr, load_s_test=load_s_test, option=option, main_option = main_option)

    
    if option == 'fair':
        with open("./influence_score/{}/{}_influence_score_seed_{}_sen_attr_{}.txt".format(main_option, _dataset, _seed, _sen_attr), "wb") as fp:
            pickle.dump(influences, fp)
    elif option == 'val_loss':
        with open("./influence_score/{}/{}_val_loss_influence_score_seed_{}_sen_attr_{}.txt".format(main_option, _dataset, _seed, _sen_attr), "wb") as fp:
            pickle.dump(influences, fp)

args = get_args()

seed = args.seed
set_seed(seed)

GPU_NUM = args.gpu

dataset = args.dataset
target = args.target
sen_attr = args.sen_attr

option = args.option
main_option = args.main_option

device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available(): torch.cuda.set_device(device)

print(device)

if dataset in ("celeba", "utkface"):
    model = torch.load("./model/{}_resnet18_target_{}_seed_{}_sen_attr_{}".format(dataset, target, seed, sen_attr))
    num_classes, num_groups, train_loader, valid_loader, test_loader = DataloaderFactory.get_dataloader(dataset, img_size=128,
                                                                                      batch_size=128, seed=100,
                                                                                      num_workers=0,
                                                                                      target=target,
                                                                                      sen_attr=sen_attr)

    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
else:
    model = torch.load("./model/{}/{}_MLP_target_{}_seed_{}_sen_attr_{}".format(main_option, dataset, target, seed, sen_attr))
    num_classes, num_groups, train_loader, valid_loader, test_loader = DataloaderFactory.get_dataloader(name=dataset, batch_size=128, seed=seed, num_workers=0, target=target, influence_scores=[], sen_attr=sen_attr)

    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    



print("train dataset number: {}".format(len(train_dataset)))

r = args.r
t = args.t

random_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
train_sampler = torch.utils.data.DataLoader(train_dataset, batch_size=t, sampler=random_sampler)

weights = torch.ones(len(train_dataset))

if args.calc_option == "grad_V":
    get_grad_V(args.constraint, valid_loader, model, dataset, seed, sen_attr, main_option, save=True)
elif args.calc_option == "s_test":
    get_avg_s_test(model, valid_loader, train_sampler, args.constraint, weights, r, dataset, seed, sen_attr, t, option, main_option, save=True)
elif args.calc_option == "influence":
    get_influence_score(model, train_loader, valid_loader, train_sampler, args.constraint, weights, dataset, seed, sen_attr, t, r, option, main_option, load_s_test=True)



