import torch
import torch.nn as nn
import numpy as np
from utils import split_dataset, exp_normalize, calc_loss_diff, calc_fairness_metric, get_error_and_violations, debias_weights, set_seed
from torch.optim import SGD, Adam
from mlp import MLP
from adult_dataloader import CustomDataset
from torch.utils.data import DataLoader
from influence_function import calc_influence_dataset
from tqdm import tqdm
from argument import get_args
from time import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def main():
    GPU_NUM = 3
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(device)

    print(device)
    args = get_args()

    seed = args.seed
    set_seed(seed)

    dataset = args.dataset
    fairness_constraint = args.constraint
    method = args.method
    epoch = args.epoch
    iteration = args.iteration
    scale_factor = args.scaler
    eta = args.eta

    print(seed)
    if dataset == "adult":
        from adult_dataloader import get_data
    elif dataset == "bank":
        from bank_dataloader import get_data
    elif dataset == "compas":
        from compas_dataloader import get_data

    X_train, y_train, X_test, y_test, protected_train, protected_test = get_data()
    print(X_train.shape, X_test.shape)

    if method == "naive_leave_k_out":
        k = args.k
        top_k_idx = np.load("./leave_k_out_idx/naive_" + str(seed) + '_' + str(dataset) + '_' + str(fairness_constraint) + "_top" + str(k) + "_idx.npy")
        X_train = np.delete(X_train, top_k_idx, axis=0)
        y_train = np.delete(y_train, top_k_idx, axis=0)
        print(X_train.shape, y_train.shape)
        print(sum(protected_train[0][top_k_idx]), sum(protected_train[1][top_k_idx]))
        protected_train[0] = np.delete(protected_train[0], top_k_idx, axis=0)
        protected_train[1] = np.delete(protected_train[1], top_k_idx, axis=0)
    elif method == "naive_leave_bottom_k_out":
        k = args.k
        bottom_k_idx = np.load("./leave_k_out_idx/naive_" + str(seed) + '_' + str(dataset) + '_' + str(fairness_constraint) + "_bottom" + str(k) + "_idx.npy")
        X_train = np.delete(X_train, bottom_k_idx, axis=0)
        y_train = np.delete(y_train, bottom_k_idx, axis=0)
        print(X_train.shape)
        print(sum(protected_train[0][bottom_k_idx]), sum(protected_train[1][bottom_k_idx]))
        protected_train[0] = np.delete(protected_train[0], bottom_k_idx, axis=0)
        protected_train[1] = np.delete(protected_train[1], bottom_k_idx, axis=0)
        print(X_train.shape)
    elif method == "leave_random_k_out":
        k = args.k
        random_k_idx = np.random.randint(0, len(X_train)-1, k)
        X_train = np.delete(X_train, random_k_idx, axis=0)
        y_train = np.delete(y_train, random_k_idx, axis=0)
        protected_train[0] = np.delete(protected_train[0], random_k_idx, axis=0)
        protected_train[1] = np.delete(protected_train[1], random_k_idx, axis=0)
        print(X_train.shape)

    X_groups_train, y_groups_train = split_dataset(X_train, y_train, protected_train)
    X_groups_test, y_groups_test = split_dataset(X_test, y_test, protected_test)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    batch_size = 128



    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    test_dataset = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    gpu = -1 if torch.cuda.is_available() else -1

    if fairness_constraint == 'eopp':
        from utils import get_eopp_idx
        get_idx = get_eopp_idx
    elif fairness_constraint == 'eo':
        from utils import get_eo_idx
        get_idx = get_eo_idx
    elif fairness_constraint == 'dp':
        from utils import get_eo_idx
        get_idx = get_eo_idx

    constraint_idx_train = get_idx(y_groups_train)
    constraint_idx_test = get_idx(y_groups_test)

    if dataset in ("compas", "adult", "bank"):
        model = MLP(
            feature_size=X_train.shape[1],
            hidden_dim=50,
            num_classes=2,
            num_layer=2
        )
    elif dataset in ("UTKFace_preprocessed"):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

    optimizer = SGD(model.parameters(), lr=0.03, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')

    multipliers = np.zeros(len(protected_train)) if (fairness_constraint == 'eopp' or fairness_constraint == 'dp') else np.zeros(len(protected_train) * 2)

    max_iter = 0
    max_tradeoff = 0
    max_tradeoff_test_acc = 0
    max_tradeoff_test_fairness_metric = 0
    max_tradeoff_trng_acc = 0
    max_tradeoff_trng_fairness_metric = 0

    naive_acc = args.naive_acc
    naive_vio = args.naive_vio

    top_k_idx = np.array([])

    for _iter in range(1, iteration + 1):
        print()
        print("Iteration: {}".format(_iter))
        if _iter == 1 or method == 'naive': weights = torch.ones(len(X_train))
        elif method == 'influence' and _iter >= 2:
            r = args.r
            t = args.t
            random_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
            train_sampler = torch.utils.data.DataLoader(train_dataset, batch_size=t, sampler=random_sampler)
            start = time()

            weights = torch.tensor(calc_influence_dataset(X_train, y_train, constraint_idx_train, X_groups_train, y_groups_train,
                                                            model, train_sampler, weights, gpu=gpu, constraint=fairness_constraint, r=r,
                                                          recursion_depth=t, scale=500.0))
            end = time()
            print("Elapsed time for calculating weights {:.1f}s".format(end-start))
            print(weights[:3], torch.mean(weights))
            weights = exp_normalize(weights, scale_factor)

        elif method == 'reweighting' and _iter >= 2:
            weights = torch.tensor(debias_weights(fairness_constraint, y_train.cpu(), protected_train, multipliers))

        elif method == 'leave_k_out_fine_tuning':
            if _iter % args.term == 0:
                r = args.r
                t = args.t
                k = args.k
                random_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
                train_sampler = torch.utils.data.DataLoader(train_dataset, batch_size=t, sampler=random_sampler)
                start = time()
                scores = np.array(
                    calc_influence_dataset(X_train, y_train, constraint_idx_train, X_groups_train, y_groups_train,
                                           model, train_sampler, weights, gpu=gpu, constraint=fairness_constraint, r=r,
                                           recursion_depth=t, scale=500.0))
                end = time()
                print("Elapsed time for calculating weights {:.1f}s".format(end - start))
                top_k_idx = np.append(top_k_idx, np.argpartition(scores, -k)[-k:]).astype(int)

                X_train, y_train, X_test, y_test, protected_train, protected_test = get_data()

                X_train = np.delete(X_train, top_k_idx, axis=0)
                y_train = np.delete(y_train, top_k_idx, axis=0)
                protected_train[0] = np.delete(protected_train[0], top_k_idx, axis=0)
                protected_train[1] = np.delete(protected_train[1], top_k_idx, axis=0)

                X_groups_train, y_groups_train = split_dataset(X_train, y_train, protected_train)
                X_groups_test, y_groups_test = split_dataset(X_test, y_test, protected_test)

                X_train = torch.FloatTensor(X_train)
                y_train = torch.LongTensor(y_train)
                X_test = torch.FloatTensor(X_test)
                y_test = torch.LongTensor(y_test)

                weights = torch.ones(len(X_train))
                print("Remove top {} data!".format(k), X_train.shape)

        print("Weights: {}".format(weights))
        for _epoch in tqdm(range(epoch)):
            model.train()
            i = 0
            for z, t, idx in train_loader:
                if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
                model.train()
                y_pred = model(z)

                weight = weights[i * batch_size: (i + 1) * batch_size] if (i + 1) * batch_size <= len(X_train) else weights[i * batch_size:]
                if torch.cuda.is_available(): weight = weight.cuda()
                if len(y_pred.shape) == 1: y_pred = y_pred.unsqueeze(0)
                loss = torch.mean(weight * criterion(y_pred, t))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1

        if method == 'reweighting':
            if torch.cuda.is_available(): model, X_train = model.cuda(), X_train.cuda()
            y_pred_train = model(X_train).argmax(dim=1).detach().cpu().numpy()
            _, violations = get_error_and_violations(fairness_constraint, y_pred_train, y_train.cpu(), protected_train)
            multipliers += eta * np.array(violations)

        model.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                X_test, y_test = X_test.cuda(), y_test.cuda()
                X_train, y_train = X_train.cuda(), y_train.cuda()
            test_accuracy = sum(model(X_test).argmax(dim=1) == y_test) / float(len(y_test))
            trng_accuracy = sum(model(X_train).argmax(dim=1) == y_train) / float(len(y_train))

        print("Iteration {}, Trng Acc: {:.2f}%".format(_iter, trng_accuracy * 100))
        print("Iteration {}, Test Acc: {:.2f}%".format(_iter, test_accuracy * 100))

        violation = calc_loss_diff(fairness_constraint, X_groups_test, y_groups_test, constraint_idx_test, model)
        print("Iteration {}, Violation: {:.4f}".format(_iter, violation))

        train_fairness_metric = calc_fairness_metric(fairness_constraint, X_groups_train, y_groups_train, model)
        test_fairness_metric = calc_fairness_metric(fairness_constraint, X_groups_test, y_groups_test, model)
        print("Iteration {}, Trng Fairness metric: {:.2f}%".format(_iter, train_fairness_metric * 100))
        print("Iteration {}, Test Fairness metric: {:.2f}%".format(_iter, test_fairness_metric * 100))

        tradeoff = (naive_vio - test_fairness_metric * 100) / (naive_acc - test_accuracy * 100)
        if tradeoff > max_tradeoff:
            max_iter = _iter
            max_tradeoff  = tradeoff
            max_tradeoff_test_acc = test_accuracy * 100
            max_tradeoff_test_fairness_metric = test_fairness_metric * 100
            max_tradeoff_trng_acc = trng_accuracy * 100
            max_tradeoff_trng_fairness_metric = train_fairness_metric * 100
        
        if str(method) == "influence" or str(method) == "reweighting":
            log = open('./log/' + str(dataset) + ' ' + str(method) + ' ' + str(fairness_constraint) + "_new.txt", 'a', encoding="UTF8")
            if test_accuracy * 100 >= naive_acc and str(method) == "reweighting":
                log.write("seed: {}, eta: {}, Trng Acc: {:.2f}, Trng Fairness Metric: {:.2f}, Test Acc: {:.2f}, Test Fairness Metric: {:.2f} \n".format(seed, eta, trng_accuracy * 100, train_fairness_metric * 100, test_accuracy * 100, test_fairness_metric * 100))
            elif test_accuracy * 100 >= naive_acc and str(method) == "influence":
                log.write("seed: {}, scaler: {}, Trng Acc: {:.2f}, Trng Fairness Metric: {:.2f}, Test Acc: {:.2f}, Test Fairness Metric: {:.2f} \n".format(seed, scale_factor, trng_accuracy * 100, train_fairness_metric * 100, test_accuracy * 100, test_fairness_metric * 100))
            log.close()
       

    log = open('./log/' + str(dataset) + ' ' + str(method) + ' ' + str(fairness_constraint) + "_new.txt", 'a', encoding="UTF8")
     
    if str(method) == "reweighting":
        log.write("seed: {}, eta: {}, Trng Acc: {:.2f}, Trng Fairness Metric: {:.2f}, Test Acc: {:.2f}, Test Fairness Metric: {:.2f}, Tradeoff: {:.4f} \n".format(seed, eta, max_tradeoff_trng_acc, max_tradeoff_trng_fairness_metric, max_tradeoff_test_acc, max_tradeoff_test_fairness_metric, max_tradeoff))
    elif str(method) == "influence":
        log.write("seed: {}, scaler: {}, Trng Acc: {:.2f}, Trng Fairness Metric: {:.2f}, Test Acc: {:.2f}, Test Fairness Metric: {:.2f}, Tradeoff: {:.4f} \n".format(seed, scale_factor, max_tradeoff_trng_acc, max_tradeoff_trng_fairness_metric, max_tradeoff_test_acc, max_tradeoff_test_fairness_metric, max_tradeoff))
    elif str(method) == "naive_leave_k_out":
        log.write(
            "seed: {}, k: {}, Trng Acc: {:.2f}, Trng Fairness Metric: {:.2f}, Test Acc: {:.2f}, Test Fairness Metric: {:.2f}, Tradeoff: {:.4f} \n".format(
                seed, k, max_tradeoff_trng_acc, max_tradeoff_trng_fairness_metric, max_tradeoff_test_acc,
                max_tradeoff_test_fairness_metric, max_tradeoff))
    elif str(method) in ["naive_leave_bottom_k_out", "leave_random_k_out"] and _iter == iteration:
        log.write(
            "seed: {}, k: {}, Trng Acc: {:.2f}, Trng Fairness Metric: {:.2f}, Test Acc: {:.2f}, Test Fairness Metric: {:.2f} \n".format(seed, k, trng_accuracy * 100, train_fairness_metric * 100, test_accuracy * 100, test_fairness_metric * 100)
        )
    log.close()

    if method == 'naive' and args.idx_save == 1:
        r = args.r
        t = args.t
        k = args.k
        random_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
        train_sampler = torch.utils.data.DataLoader(train_dataset, batch_size=t, sampler=random_sampler)
        influence_scores = np.array(calc_influence_dataset(X_train, y_train, constraint_idx_train, X_groups_train, y_groups_train,
                                   model, train_sampler, weights, gpu=gpu, constraint=fairness_constraint, r=r,
                                   recursion_depth=t, scale=500.0))

        largest_idx = np.argpartition(influence_scores, -k)[-k:]
        smallest_idx = np.argpartition(influence_scores, k)[:k]
        np.save("./leave_k_out_idx/naive_" + str(seed) + '_' + str(dataset)  + '_' + str(fairness_constraint) + "_top" + str(k) + "_idx", largest_idx)
        np.save("./leave_k_out_idx/naive_" + str(seed) + '_' + str(dataset)  + '_' + str(fairness_constraint) + "_bottom" + str(k) + "_idx", smallest_idx)

    if args.model_save == 1:
        if str(method) == "naive_leave_k_out":
            torch.save(model, "./model/{}_{}_{}_{}_{}".format(str(seed), str(dataset), str(method), str(fairness_constraint), str(k)))
        print("Model Save Done!")
if __name__ == '__main__':
    main()

