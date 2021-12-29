import torch
from data_handler.dataloader_factory import DataloaderFactory
import numpy as np
from influence_function_image import avg_s_test, s_test, grad_V
import time

GPU_NUM = 3
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

print(device)

model = torch.load("./model/celeba_resnet18_target_young")

num_classes, num_groups, train_loader, test_loader = DataloaderFactory.get_dataloader("celeba", img_size=128,
                                                                                      batch_size=128, seed=100,
                                                                                      num_workers=4,
                                                                                      target='Young')

train_dataset = train_loader.dataset
test_dataset = test_loader.dataset

print("train dataset number: {}".format(len(train_dataset)))

r = 450
t = 400

random_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
train_sampler = torch.utils.data.DataLoader(train_dataset, batch_size=t, sampler=random_sampler)

weights = torch.ones(len(train_dataset))

#print("start")
#grad_V('eopp', train_loader, model, save=True)
'''
print(s_test(model=model, dataloader=train_loader, random_sampler=train_sampler, constraint='eopp', weights=weights, recursion_depth=t, load_gradV=True, save=True))
'''
#print("end")
print("start")
print(avg_s_test(model = model, dataloader= train_loader, random_sampler=train_sampler, constraint='eopp', weights=weights, recursion_depth=t, r = r))

