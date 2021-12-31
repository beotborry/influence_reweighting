import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD, Adam
from data_handler.dataloader_factory import DataloaderFactory
from utils import set_seed, get_accuracy
from argument import get_args
from tqdm import tqdm
from influence_function_image import grad_V, s_test, calc_influence

def main():
    GPU_NUM = 3
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(device)
    print(device)

    #torch.backends.cudnn.benchmark = True

############### parisng argument ##########################
    args = get_args()
    seed = args.seed
    set_seed(seed)

    dataset = args.dataset
    fairness_constraint = args.constraint
    method = args.method
    epoch = args.epoch
    target = args.target
###########################################################

    num_classes, num_groups, train_loader, test_loader = DataloaderFactory.get_dataloader(dataset, img_size=128,
                                                                                          batch_size=128, seed=seed,
                                                                                          num_workers=4,
                                                                                          target=target)

    # X, feature, group, target, (idx, img_name) = dataloader
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
        
    ############################ model train #######################################
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    #optimizer = SGD(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    random_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
    train_sampler = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=random_sampler)

    weights = torch.ones(len(train_dataset))

    best_acc = 0.0
    for _epoch in tqdm(range(epoch)):
        model.train()
        i = 0

        for z, _, _, t, _ in tqdm(train_loader):
            if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
            y_pred = model(z)

            loss = torch.mean(criterion(y_pred, t))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
        
        model.eval()
        test_acc = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                z, _, _, t, _ = data
                if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
                y_pred = model(z)

                test_acc += get_accuracy(y_pred, t)

            test_acc = test_acc / i


        print('Test Accuracy: {:2f}'.format(test_acc * 100))
        if test_acc * 100 >= best_acc:
            print('Test Accuracy: {:2f}, Model Save!'.format(test_acc * 100))
            torch.save(model, './model/{}_resnet18_target_{}'.format(dataset, target))
            #torch.save(model, './model/celeba_resnet18_target_young')
            best_acc = test_acc * 100

    

if __name__ == '__main__':
    main()
