import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD, Adam, AdamW
from data_handler.dataloader_factory import DataloaderFactory
from utils import set_seed, get_accuracy
from argument import get_args
from tqdm import tqdm
from influence_function_image import grad_V, s_test, calc_influence
from utils_image import compute_confusion_matrix, calc_fairness_metric
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from mlp import MLP


def main():
############### parisng argument ##########################
    train_dataset_length = {'adult':28941, 'compas': 3946, 'bank': 19512}
    
    args = get_args()
    seed = args.seed
    set_seed(seed)

    GPU_NUM = args.gpu

    dataset = args.dataset
    fairness_constraint = args.constraint
    method = args.method
    epoch = args.epoch
    target = args.target
    sen_attr = args.sen_attr
    fine_tuning = args.fine_tuning

    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(device)
    print(device)

    if method == 'naive':
        num_classes, num_groups, train_loader, valid_loader, test_loader = DataloaderFactory.get_dataloader(name=dataset,
                                                                                                    batch_size=128,
                                                                                                    seed=seed,
                                                                                                    num_workers=0,
                                                                                                    target=target,
                                                                                                    influence_scores=[],
                                                                                                    sen_attr=sen_attr)
    elif method == 'naive_leave_k_out' or method == 'naive_leave_bottom_k_out':
        k = args.k
        #print("k: {}".format(k))
        print("{} percent".format(k))

        num_classes, num_groups, train_loader, valid_loader, test_loader = DataloaderFactory.get_dataloader(name=dataset,
                                                                                                    batch_size=128,
                                                                                                    seed=seed,
                                                                                                    num_workers=0,
                                                                                                    target=target,
                                                                                                    influence_scores=[],
                                                                                                    sen_attr=sen_attr)

        with open("./influence_score/{}_influence_score_seed_{}_sen_attr_{}.txt".format(dataset, seed, sen_attr), "rb") as fp:
            influences = np.array(pickle.load(fp))

        pivot = int(train_dataset_length[dataset] * (k / 100.0))
        if method == 'naive_leave_k_out':
            with open("./influence_score/{}_val_loss_influence_score_seed_{}_sen_attr_{}.txt".format(dataset, seed, sen_attr), "rb") as fp:
                influences_val_loss = np.array(pickle.load(fp))

            fair_top = np.argpartition(influences, -pivot)[-pivot:]
            val_loss_top = np.argpartition(influences_val_loss, -pivot)[-pivot:]
            remove_idx = np.intersect1d(fair_top, val_loss_top)
            #remove_idx = np.argpartition(influences, -pivot)[-pivot:]
        elif method == 'naive_leave_bottom_k_out':
            with open("./influence_score/{}_val_loss_influence_score_seed_{}_sen_attr_{}.txt".format(dataset, seed, sen_attr), "rb") as fp:
                influences_val_loss = np.array(pickle.load(fp))

            fair_bottom = np.argpartition(influences, pivot)[:pivot]
            val_loss_bottom = np.argpartition(influences_val_loss, pivot)[:pivot]
            remove_idx = np.intersect1d(fair_bottom, val_loss_bottom)
            #remove_idx = np.argpartition(influences, pivot)[:pivot]

        removed_data_log = np.zeros((num_groups, num_classes))
        for idx in remove_idx:
            _, _, g, l, _ = train_loader.dataset[idx]
            removed_data_log[int(g), int(l)] += 1

        print(removed_data_log)    
        
        _, _, train_loader, _, _ = DataloaderFactory.get_dataloader(name=dataset,
                                                                    batch_size=128,
                                                                    seed=seed,
                                                                    num_workers=0,
                                                                    target=target,
                                                                    influence_scores=remove_idx,
                                                                    sen_attr=sen_attr)

    test_acc_arr = []
    test_fairness_metric_arr = []
    trng_acc_arr = []
    trng_fairness_metric_arr = []
    valid_acc_arr = []
    valid_fairness_metric_arr = []

    confu_mat_arr = []

    if dataset not in ('adult', 'compas', 'bank'):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    else:
        if dataset == 'adult': feature_size = 97
        elif dataset == 'compas': feature_size = 400
        elif dataset == 'bank': feature_size = 56

        if fine_tuning == 0:
            model = MLP(
                feature_size=feature_size,
                hidden_dim=50,
                num_classes=2,
                num_layer=2
            )
        elif fine_tuning == 1:
            model = torch.load("./model/{}_MLP_target_{}_seed_{}_sen_attr_{}".format(dataset, target, seed, sen_attr))
        #optimizer = SGD(model.parameters(), lr=0.03, weight_decay=5e-4)
        optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss(reduction='none')

    if method == 'naive':

        # scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=2)
        # scheduler = CosineAnnealingLR(optimizer, T_max=10, verbose=True)

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
            trng_acc_arr.append(trng_acc)

            print('Trng Accuracy: {:.2f}'.format(trng_acc * 100))
            confu_mat_train = compute_confusion_matrix(train_loader, model)
            trng_fair = calc_fairness_metric(args.constraint, confu_mat_train)
            trng_fairness_metric_arr.append(trng_fair)
            model.eval()

            valid_acc = 0.0
            with torch.no_grad():
                for i, data in enumerate(valid_loader):
                    z, _, _, t, _ = data
                    if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
                    y_pred = model(z)

                    valid_acc += torch.sum(get_accuracy(y_pred, t, reduction='none')).item()

                valid_acc = valid_acc / len(valid_loader.dataset)
                valid_acc_arr.append(valid_acc)

            print('Valid Accuracy: {:.2f}'.format(valid_acc * 100))

            confu_mat_valid = compute_confusion_matrix(valid_loader, model)
            valid_fair = calc_fairness_metric(args.constraint, confu_mat_valid)
            valid_fairness_metric_arr.append(valid_fair)

            test_acc = 0.0

            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    z, _, _, t, _ = data
                    if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
                    y_pred = model(z)

                    test_acc += torch.sum(get_accuracy(y_pred, t, reduction='none')).item()

                test_acc = test_acc / len(test_loader.dataset)
                test_acc_arr.append(test_acc)

            print('Test Accuracy: {:.2f}'.format(test_acc * 100))

            confu_mat_test = compute_confusion_matrix(test_loader, model)
            test_fair = calc_fairness_metric(args.constraint, confu_mat_test)
            test_fairness_metric_arr.append(test_fair)

            scheduler.step(test_acc)

            if test_acc * 100 >= best_acc:
                print('Test Accuracy: {:.2f}, Model Save!'.format(test_acc * 100))
                # torch.save(model.state_dict(), './model/{}_resnet18_target_{}_seed_{}'.format(dataset, target, seed))
                if dataset not in ('adult', 'compas', 'bank'): torch.save(model, './model/{}_resnet18_target_{}_seed_{}_sen_attr_{}'.format(dataset, target, seed, sen_attr))
                else: torch.save(model, './model/{}_MLP_target_{}_seed_{}_sen_attr_{}'.format(dataset, target, seed, sen_attr))

                best_acc = test_acc * 100

            # scheduler.step()

        log_arr = [trng_acc_arr, trng_fairness_metric_arr, valid_acc_arr, valid_fairness_metric_arr, test_acc_arr,
                   test_fairness_metric_arr]

        with open("./log/{}_seed_{}_sen_attr_{}_naive_log.txt".format(dataset, seed, sen_attr), "wb") as fp:
            pickle.dump(log_arr, fp)

    elif method == "naive_leave_k_out":
        trng_acc = 0.0

        for _ in tqdm(range(epoch)):
            model.train()
            i = 0

            for z, _, _, t, tup in tqdm(train_loader):
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

            confu_mat_train = [confu_mat_train['0'], confu_mat_train['1']]
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
                
            confu_mat_test = compute_confusion_matrix(test_loader, model)
            test_fairness_metric = calc_fairness_metric("eopp", confu_mat_test)

            confu_mat_test = [confu_mat_test['0'], confu_mat_test['1']]
            print("Test Acc: {:.2f}, Test fairness metric: {:.2f}".format(test_acc * 100, test_fairness_metric * 100))
            scheduler.step(test_acc)

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

            confu_mat_valid = compute_confusion_matrix(valid_loader, model)
            valid_fairness_metric = calc_fairness_metric("eopp", confu_mat_valid)

            confu_mat_valid = [confu_mat_valid['0'], confu_mat_valid['1']]
            print("Valid Acc: {:.2f}, Valid fairness metric: {:.2f}".format(valid_acc * 100, valid_fairness_metric * 100))

            valid_acc_arr.append(valid_acc.item())
            valid_fairness_metric_arr.append(valid_fairness_metric)

            confu_mat_arr.append([confu_mat_train, confu_mat_valid, confu_mat_test])
            # scheduler.step(test_acc)

        acc_fair_log_arr = [trng_acc_arr, trng_fairness_metric_arr, valid_acc_arr, valid_fairness_metric_arr, test_acc_arr, test_fairness_metric_arr]

        with open("./log/{}_seed_{}_k_{}_sen_attr_{}_acc_fair_log.txt".format(dataset, seed, k,sen_attr), "wb") as fp:
            pickle.dump(acc_fair_log_arr, fp)

        with open("./log/{}_seed_{}_k_{}_sen_attr_{}_removed_data_info.txt".format(dataset, seed, k, sen_attr), "wb") as fp:
            pickle.dump(removed_data_log, fp)
        
        with open("./log/{}_seed_{}_k_{}_sen_attr_{}_confusion_matrix.txt".format(dataset, seed, k, sen_attr), "wb") as fp:
            pickle.dump(confu_mat_arr, fp)
    
    elif method == "naive_leave_bottom_k_out":
        trng_acc = 0.0

        for _ in tqdm(range(epoch)):
            model.train()
            i = 0

            for z, _, _, t, tup in tqdm(train_loader):
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
            
            confu_mat_train = [confu_mat_train['0'], confu_mat_train['1']]
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
                
            confu_mat_test = compute_confusion_matrix(test_loader, model)
            test_fairness_metric = calc_fairness_metric("eopp", confu_mat_test)

            confu_mat_test = [confu_mat_test['0'], confu_mat_test['1']]
            print("Test Acc: {:.2f}, Test fairness metric: {:.2f}".format(test_acc * 100, test_fairness_metric * 100))
            scheduler.step(test_acc)

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

            confu_mat_valid = compute_confusion_matrix(valid_loader, model)
            valid_fairness_metric = calc_fairness_metric("eopp", confu_mat_valid)

            confu_mat_valid = [confu_mat_valid['0'], confu_mat_valid['1']]
            print("Valid Acc: {:.2f}, Valid fairness metric: {:.2f}".format(valid_acc * 100, valid_fairness_metric * 100))

            valid_acc_arr.append(valid_acc.item())
            valid_fairness_metric_arr.append(valid_fairness_metric)

            confu_mat_arr.append([confu_mat_train, confu_mat_valid, confu_mat_test])

            # scheduler.step(test_acc)

        acc_fair_log_arr = [trng_acc_arr, trng_fairness_metric_arr, valid_acc_arr, valid_fairness_metric_arr, test_acc_arr, test_fairness_metric_arr]

        with open("./log/{}_seed_{}_bottom_k_{}_sen_attr_{}_acc_fair_log.txt".format(dataset, seed, k, sen_attr), "wb") as fp:
            pickle.dump(acc_fair_log_arr, fp)

        with open("./log/{}_seed_{}_bottom_k_{}_sen_attr_{}_removed_data_info.txt".format(dataset, seed, k, sen_attr), "wb") as fp:
            pickle.dump(removed_data_log, fp)
        
        with open("./log/{}_seed_{}_bottom_k_{}_sen_attr_{}_confusion_matrix.txt".format(dataset, seed, k, sen_attr), "wb") as fp:
            pickle.dump(confu_mat_arr, fp)

if __name__ == '__main__':
    main()
