import torch
from data_handler.dataloader_factory import DataloaderFactory
from influence_function_image import avg_s_test, grad_V, calc_influence_dataset
import pickle
from argument_influence_image import get_args
from utils import set_seed
from torch.utils.data import DataLoader


def get_grad_V(constraint, dataloader, model, _dataset, _seed, _sen_attr, save=True):
    grad_V(constraint, dataloader, model, _dataset, _seed, _sen_attr, save=save)

def get_avg_s_test(model, dataloader, random_sampler, constraint, weights, r, _dataset, _seed, _sen_attr, recursion_depth, save=True):
    avg_s_test(model=model, dataloader=dataloader, random_sampler=random_sampler, constraint=constraint, weights=weights, r=r, recursion_depth=recursion_depth, _dataset=_dataset, _seed=_seed, _sen_attr=_sen_attr, save=save)

def get_influence_score(model, dataloader, s_test_dataloader, random_sampler, constraint, weights, _dataset, _seed, _sen_attr, recursion_depth, r, load_s_test=True):
    influences = calc_influence_dataset(model=model, dataloader=dataloader, s_test_dataloader = s_test_dataloader, random_sampler=random_sampler, constraint=constraint, weights=weights, recursion_depth=recursion_depth, r=r, _dataset=_dataset, _seed=_seed, _sen_attr=_sen_attr, load_s_test=load_s_test)

    with open("./influence_score/{}_influence_score_seed_{}_sen_attr_{}.txt".format(_dataset, _seed, sen_attr), "wb") as fp:
        pickle.dump(influences, fp)

args = get_args()

seed = args.seed
set_seed(seed)

GPU_NUM = args.gpu

dataset = args.dataset
target = args.target
sen_attr = args.sen_attr

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
    model = torch.load("./model/{}_MLP_target_{}_seed_{}_sen_attr_{}".format(dataset, target, seed, sen_attr))
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
    get_grad_V(args.constraint, valid_loader, model, dataset, seed, sen_attr, save=True)
elif args.calc_option == "s_test":
    get_avg_s_test(model, valid_loader, train_sampler, args.constraint, weights, r, dataset, seed, sen_attr, t, save=True)
elif args.calc_option == "influence":
    get_influence_score(model, train_loader, valid_loader, train_sampler, args.constraint, weights, dataset, seed, sen_attr, t, r, load_s_test=True)



