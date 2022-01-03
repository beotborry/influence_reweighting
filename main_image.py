import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD, Adam
from data_handler.dataloader_factory import DataloaderFactory
from utils import set_seed, get_accuracy
from argument import get_args
from tqdm import tqdm
from influence_function_image import grad_V, s_test, calc_influence
from utils_image import compute_confusion_matrix, calc_fairness_metric
import pickle


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
    if method == 'naive':
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
            for z, _, _, t, _ in tqdm(train_loader):
                if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
                y_pred = model(z)

                loss = torch.mean(criterion(y_pred, t))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
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
                best_acc = test_acc * 100

    elif method == 'naive_leave_k_out':
        test_acc_arr = []
        test_fairness_metric_arr = []
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        k = args.k

        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(reduction='none')

        with open("./{}_influence_score_seed_{}.txt".format(dataset, seed), "rb") as fp:
            influences = np.array(pickle.load(fp))

        top_k_idx =  np.argpartition(influences, -k)[-k:]
        print(top_k_idx)

        best_acc = 0.0
        best_fairness_metric = 100.0

        for _epoch in tqdm(range(epoch)):
            model.train()
            i = 0

            for z, _, _, t, tup in tqdm(train_loader):
                idx = tup[0]
                remove_idx = np.where(np.in1d(idx.numpy(), top_k_idx))[0]
                
                if len(remove_idx) != 0: 
                    print("removed {} data!".format(len(remove_idx)))
                    z = torch.tensor(np.delete(z.numpy(), remove_idx, axis = 0))
                    t = torch.tensor(np.delete(t.numpy(), remove_idx, axis = 0))

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
                test_acc /= i
                
            confu_mat = compute_confusion_matrix(test_loader, model)
            test_fairness_metric = calc_fairness_metric("eopp", confu_mat)


            print("Test Acc: {:.2f}, Test fairness metric: {:.2f}".format(test_acc * 100, test_fairness_metric * 100))

            test_acc_arr.append(test_acc)
            test_fairness_metric_arr.append(test_fairness_metric)

        log_arr = [test_acc_arr, test_fairness_metric_arr]

        with open("./{}_seed_{}_k_{}_log.txt".format(dataset, seed, k)) as fp:
            pickle.dump(log_arr, fp)

if __name__ == '__main__':
    main()
