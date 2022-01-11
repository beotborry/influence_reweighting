import torch
import pickle
import numpy as np
import torch.nn as nn
from torch.optim import SGD
from argument import get_args
from utils import set_seed, get_accuracy
from utils_image import compute_confusion_matrix, calc_fairness_metric
from torch.utils.data import DataLoader
from mlp import MLP
from tqdm import tqdm

def main():
    args = get_args()
    seed = args.seed
    set_seed(seed)

    GPU_NUM = args.gpu

    dataset = args.dataset
    fairness_constraint = args.constraint
    method = args.method
    epoch = args.epoch
    target = args.target

    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(device)
    print(device)

    if dataset == "adult":
        from adult_dataloader import get_data
        from adult_dataloader import Adult as CustomDataset

    _X_train, _y_train, X_test, y_test, _protected_train, protected_test = get_data()
    
    pivot = int(len(_X_train) * 0.8)
    X_train = _X_train[:pivot]
    y_train = _y_train[:pivot]
    protected_train = [_protected_train[0][:pivot], _protected_train[1][:pivot]]
    
    X_valid = _X_train[pivot:]
    y_valid = _y_train[pivot:]
    protected_valid = [_protected_train[0][pivot:], _protected_train[1][pivot:]]

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_valid = torch.FloatTensor(X_valid)
    y_valid = torch.LongTensor(y_valid)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    train_dataset = CustomDataset(X_train, y_train, protected_train)
    valid_dataset = CustomDataset(X_valid, y_valid, protected_valid)
    test_dataset = CustomDataset(X_test, y_test, protected_test)

    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle= False, num_workers = 0)
    valid_loader = DataLoader(valid_dataset, batch_size = 128, shuffle=False, num_workers = 0)
    test_loader = DataLoader(test_dataset, batch_size = 128, shuffle=False, num_workers = 0)

    if method == "naive":
        model = MLP(
            feature_size = X_train.shape[1],
            hidden_dim = 50,
            num_classes = 2,
            num_layer = 2
        )

        optimizer = SGD(model.parameters(), lr = 0.03, weight_decay = 5e-4)
        criterion = nn.CrossEntropyLoss(reduction='none')

        best_acc = 0.0
        trng_acc = 0.0
        for _epoch in tqdm(range(epoch)):
            model.train()
            for z, _, _, t, _ in tqdm(train_loader):
                if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
                y_pred = model(z)

                loss = torch.mean(criterion(y_pred, t))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                trng_acc += torch.sum(get_accuracy(y_pred, t, reduction='none')).item()

            trng_acc /= len(train_loader.dataset)

            print("Trng Acc: {:.2f}".format(trng_acc * 100))

            model.eval()
            test_acc = 0.0

            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    z, _, _, t, _ = data
                    if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
                    y_pred = model(z)

                    test_acc += torch.sum(get_accuracy(y_pred, t, reduction='none'))

                test_acc = test_acc / len(test_loader.dataset)

            print("Test Acc: {:.2f}".format(test_acc * 100))

            if test_acc * 100 >= best_acc:
                print("test Acc: {:.2f}, Model Save!".format(test_acc * 100))
                torch.save(model, "./model/{}_MLP_target_{}_seed_{}".format(dataset, target, seed))
                best_acc = test_acc * 100

    elif method == "naive_leave_k_out":
        test_acc_arr = []
        test_fairness_metric_arr = []
        trng_acc_arr = []
        trng_fairness_metric_arr = []
        valid_acc_arr = []
        valid_fairness_metric_arr = []

        model = MLP(
            feature_size = X_train.shape[1],
            hidden_dim = 50,
            num_classes = 2,
            num_layer = 2
        ) 
        k = args.k
        print("k: {}".format(k))

        optimizer = SGD(model.parameters(), lr = 0.03, weight_decay = 5e-4)
        criterion = nn.CrossEntropyLoss(reduction='none')
      

        with open("./{}_influence_score_seed_{}.txt".format(dataset, seed), "rb") as fp:
            influences = np.array(pickle.load(fp))
            
        top_k_idx =  np.argpartition(influences, -k)[-k:]
        print(np.array(np.sort(top_k_idx)))
        #print(influences[top_k_idx])

        best_acc = 0.0
        best_fairness_metric = 100.0

        trng_acc = 0.0

        for _epoch in tqdm(range(epoch)):
            model.train()
            i = 0

            remove_num = 0

            for z, _, _, t, tup in tqdm(train_loader):
                idx = tup[0]
                remove_idx = np.where(np.in1d(idx.numpy(), top_k_idx))[0]
                remove_num += len(remove_idx)
                
                if len(remove_idx) != 0: 
                    #print("removed {} data!".format(len(remove_idx)))
                    z = torch.tensor(np.delete(z.numpy(), remove_idx, axis = 0))
                    t = torch.tensor(np.delete(t.numpy(), remove_idx, axis = 0))

                if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
               
                y_pred = model(z)

                loss = torch.mean(criterion(y_pred, t))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                trng_acc += torch.sum(get_accuracy(y_pred, t, reduction='none')).item()

                i += 1
            
            trng_acc /= len(train_loader.dataset)
            trng_acc_arr.append(trng_acc)

            confu_mat_train = compute_confusion_matrix(train_loader, model)
            trng_fairness_metric = calc_fairness_metric("eopp", confu_mat_train)
            trng_fairness_metric_arr.append(trng_fairness_metric)
            print("Removed total {} data!".format(remove_num))
            print("Trng Acc {:.2f}, Trng fairness metric: {:.2f}".format(trng_acc * 100, trng_fairness_metric * 100))

            model.eval()
            test_acc = 0.0
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    z, _, _, t, _ = data
                    if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
                    y_pred = model(z)

                    test_acc += torch.sum(get_accuracy(y_pred, t, reduction='none'))
                test_acc /= len(test_loader.dataset)
                
            confu_mat = compute_confusion_matrix(test_loader, model)
            test_fairness_metric = calc_fairness_metric("eopp", confu_mat)


            print("Test Acc: {:.2f}, Test fairness metric: {:.2f}".format(test_acc * 100, test_fairness_metric * 100))

            test_acc_arr.append(test_acc.item())
            test_fairness_metric_arr.append(test_fairness_metric)

            valid_acc = 0.0
            with torch.no_grad():
                for i, data in enumerate(valid_loader):
                    z, _, _, t, _ = data
                    if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
                    y_pred = model(z)

                    valid_acc += torch.sum(get_accuracy(y_pred, t, reduction='none'))
                valid_acc /= len(valid_loader.dataset)

            confu_mat = compute_confusion_matrix(valid_loader, model)
            valid_fairness_metric = calc_fairness_metric("eopp", confu_mat)

            print("Valid Acc: {:.2f}, Valid fairness metric: {:.2f}".format(valid_acc * 100, valid_fairness_metric * 100))

            valid_acc_arr.append(valid_acc.item())
            valid_fairness_metric_arr.append(valid_fairness_metric)

        log_arr = [trng_acc_arr, trng_fairness_metric_arr, valid_acc_arr, valid_fairness_metric_arr, test_acc_arr, test_fairness_metric_arr]

        with open("./image_log/{}_seed_{}_k_{}_log.txt".format(dataset, seed, k), "wb") as fp:
            pickle.dump(log_arr, fp)

    elif method == "naive_leave_bottom_k_out":
        test_acc_arr = []
        test_fairness_metric_arr = []
        trng_acc_arr = []
        trng_fairness_metric_arr = []
        valid_acc_arr = []
        valid_fairness_metric_arr = []

        model = MLP(
            feature_size = X_train.shape[1],
            hidden_dim = 50,
            num_classes = 2,
            num_layer = 2
        ) 
        k = args.k
        print("k: {}".format(k))

        optimizer = SGD(model.parameters(), lr = 0.03, weight_decay = 5e-4)
        criterion = nn.CrossEntropyLoss(reduction='none')

        with open("./{}_influence_score_seed_{}.txt".format(dataset, seed), "rb") as fp:
            influences = np.array(pickle.load(fp))

        bottom_k_idx =  np.argpartition(influences, k)[:k]
        #print(influences[bottom_k_idx])

        best_acc = 0.0
        best_fairness_metric = 100.0

        trng_acc = 0.0

        for _epoch in tqdm(range(epoch)):
            model.train()
            i = 0

            for z, _, _, t, tup in tqdm(train_loader):
                idx = tup[0]
                remove_idx = np.where(np.in1d(idx.numpy(), bottom_k_idx))[0]
                
                if len(remove_idx) != 0: 
                    #print("removed {} data!".format(len(remove_idx)))
                    z = torch.tensor(np.delete(z.numpy(), remove_idx, axis = 0))
                    t = torch.tensor(np.delete(t.numpy(), remove_idx, axis = 0))

                if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
               
                y_pred = model(z)

                loss = torch.mean(criterion(y_pred, t))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                trng_acc += torch.sum(get_accuracy(y_pred, t, reduction = 'none')).item()

                i += 1

            trng_acc /= len(train_loader.dataset)
            trng_acc_arr.append(trng_acc)

            trng_confu_mat = compute_confusion_matrix(train_loader, model)
            trng_fairness_metric = calc_fairness_metric("eopp", trng_confu_mat)
            trng_fairness_metric_arr.append(trng_fairness_metric)

            print("Trng Acc: {:.2f}, Trng fairness metric: {:.2f}".format(trng_acc * 100, trng_fairness_metric * 100))

            model.eval()
            test_acc = 0.0
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    z, _, _, t, _ = data
                    if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
                    y_pred = model(z)

                    test_acc += torch.sum(get_accuracy(y_pred, t, reduction='none'))
                test_acc /= len(test_loader.dataset)
                
            confu_mat = compute_confusion_matrix(test_loader, model)
            test_fairness_metric = calc_fairness_metric("eopp", confu_mat)


            print("Test Acc: {:.2f}, Test fairness metric: {:.2f}".format(test_acc * 100, test_fairness_metric * 100))

            test_acc_arr.append(test_acc.item())
            test_fairness_metric_arr.append(test_fairness_metric)

            valid_acc = 0.0
            with torch.no_grad():
                for i, data in enumerate(valid_loader):
                    z, _, _, t, _ = data
                    if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
                    y_pred = model(z)

                    valid_acc += torch.sum(get_accuracy(y_pred, t, reduction='none'))
                valid_acc /= len(valid_loader.dataset)

            confu_mat = compute_confusion_matrix(valid_loader, model)
            valid_fairness_metric = calc_fairness_metric("eopp", confu_mat)

            print("Valid Acc: {:.2f}, Valid fairness metric: {:.2f}".format(valid_acc * 100, valid_fairness_metric * 100))

            valid_acc_arr.append(valid_acc.item())
            valid_fairness_metric_arr.append(valid_fairness_metric)

        log_arr = [trng_acc_arr, trng_fairness_metric_arr, valid_acc_arr, valid_fairness_metric_arr, test_acc_arr, test_fairness_metric_arr]

        with open("./image_log/{}_seed_{}_bottom_k_{}_log.txt".format(dataset, seed, k), "wb") as fp:
            pickle.dump(log_arr, fp)

if __name__ == '__main__':
    main()
