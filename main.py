import torch
import torch.nn as nn
import numpy as np
from utils import split_dataset, exp_normalize, calc_loss_diff, calc_fairness_metric, get_error_and_violations, debias_weights
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

def main():
    args = get_args()

    dataset = args.dataset
    fairness_constraint = args.constraint
    method = args.method
    epoch = args.epoch
    iteration = args.iteration
    scale_factor = args.scaler
    eta = args.eta

    if dataset == "adult":
        from adult_dataloader import get_data
    elif dataset == "bank":
        from bank_dataloader import get_data
    elif dataset == "compas":
        from compas_dataloader import get_data

    X_train, y_train, X_test, y_test, protected_train, protected_test = get_data()
    print(X_train.shape, X_test.shape)

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

    optimizer = SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduction='none')

    multipliers = np.zeros(len(protected_train)) if (fairness_constraint == 'eopp' or fairness_constraint == 'dp') else np.zeros(len(protected_train) * 2)

    max_iter = 0
    max_tradeoff = 0
    max_tradeoff_acc = 0
    max_tradeoff_fairness_metric = 0

    naive_acc = 68.7
    naive_vio = 14.68

    skip = False

    for _iter in range(1, iteration + 1):
        print()
        print("Iteration: {}".format(_iter))
        if _iter == 1 or method == 'naive': weights = torch.ones(len(X_train))
        elif method == 'influence' and _iter >= 2:
            r = 10
            t = 1000
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
            weights = torch.tensor(debias_weights(fairness_constraint, y_train, protected_train, multipliers))

        print("Weights: {}".format(weights))
        for _epoch in tqdm(range(epoch)):
            model.train()
            i = 0
            for z, t, idx in train_loader:
                if gpu >= 0: z, t, model = z.cuda(), t.cuda(), model.cuda()
                model.train()
                y_pred = model(z)

                weight = weights[i * batch_size: (i + 1) * batch_size] if (i + 1) * batch_size <= len(X_train) else weights[i * batch_size:]
                if gpu >= 0: weight = weight.cuda()

                loss = torch.mean(weight * criterion(y_pred, t))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1

        if method == 'reweighting':
            y_pred_train = model(X_train).argmax(dim=1).detach().numpy()
            _, violations = get_error_and_violations(fairness_constraint, y_pred_train, y_train, protected_train)
            multipliers += eta * np.array(violations)

        model.eval()
        with torch.no_grad():
            if gpu >= 0: X_test, y_test = X_test.cuda(), y_test.cuda()
            accuracy = sum(model(X_test).argmax(dim=1) == y_test) / float(len(y_test))

        print("Iteration {}, Test Acc: {:.2f}%".format(_iter, accuracy * 100))

        violation = calc_loss_diff(fairness_constraint, X_groups_test, y_groups_test, constraint_idx_test, model)
        print("Iteration {}, Violation: {:.4f}".format(_iter, violation))

        fairness_metric = calc_fairness_metric(fairness_constraint, X_groups_test, y_groups_test, model)
        print("Iteration {}, Fairness metric: {:.2f}%".format(_iter, fairness_metric * 100))

        tradeoff = (naive_vio - fairness_metric * 100) / (naive_acc - accuracy * 100)
        if tradeoff > max_tradeoff:
            max_iter = _iter
            max_tradeoff  = tradeoff
            max_tradeoff_acc = accuracy * 100
            max_tradeoff_fairness_metric = fairness_metric * 100

    log = open(str(dataset) + ' ' + str(method) + ' ' + str(fairness_constraint) + ".txt", 'a', encoding="UTF8")
    if str(method) == "reweighting":
        log.write("eta: {}, Acc: {:.2f}, Fairness Metric: {:.2f}, Tradeoff: {:.4f} \n".format(eta, max_tradeoff_acc, max_tradeoff_fairness_metric, max_tradeoff))
    elif str(method) == "influence":
        log.write("scale factor: {}, Acc: {:.2f}, Fairness Metric: {:.2f}, Tradeoff: {:.4f} \n".format(scale_factor, max_tradeoff_acc, max_tradeoff_fairness_metric, max_tradeoff))
    log.close()

    if method == 'naive':
        influence_scores = np.array(calc_influence_dataset(X_train, y_train, constraint_idx_train, X_groups_train, y_groups_train,
                                                            model, train_loader, gpu=gpu, constraint=fairness_constraint))

        k = 100

        pos_idx = np.where(influence_scores > np.median(influence_scores))
        #largest_idx = np.argpartition(influence_scores, -k)[-k:]
        #smallest_idx = np.argpartition(influence_scores, k)[:k]

        #pos_idx = np.where(influence_scores > 0)[0]
        #neg_idx = np.where(influence_scores < 0)[0]

        tsne_model = TSNE(n_components=2, perplexity=30, learning_rate=200)

        transformed = tsne_model.fit_transform(X_train)

        transformed_largest = transformed[largest_idx]
        transformed_smallest = transformed[smallest_idx]

        #transformed_pos = transformed[pos_idx]
        #transformed_neg = transformed[neg_idx]

        xs = np.concatenate((transformed_largest[:, 0], transformed_smallest[:, 0]), axis=0)
        ys = np.concatenate((transformed_largest[:, 1], transformed_smallest[:, 1]), axis=0)
        #xs = transformed[:, 0]
        #ys = transformed[:, 1]

        is_harmful = np.concatenate((np.ones(k), np.zeros(k)), axis=0)
        #is_harmful = np.concatenate((np.ones(len(transformed_pos)), np.zeros(len(transformed_neg))))
        plt.scatter(xs, ys, c=is_harmful)
        plt.show()

if __name__ == '__main__':
    main()



