import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD, Adam, AdamW
from data_handler.dataloader_factory import DataloaderFactory
from utils import set_seed, get_accuracy, make_log_name
from argument import get_args
from tqdm import tqdm
from utils_image import compute_confusion_matrix, calc_fairness_metric
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from mlp import MLP
from shufflenet import shufflenet_v2_x1_0

def main():
############### parisng argument ##########################
    train_dataset_length = {'adult':28941, 'compas': 3946, 'bank': 19512, 'retiring_adult': 925247, 'retiring_adult_coverage': 611640}
    tabular_dataset = ['adult', 'compas', 'bank', 'retiring_adult', 'retiring_adult_coverage']
    
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
    option = args.main_option
    log_option = args.log_option
    alpha = args.alpha

    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(device)
    print(device)

    log_epi = make_log_name(args)

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

        with open("./influence_score/fair_only/{}_{}_influence_score_seed_{}_sen_attr_{}.txt".format(dataset, fairness_constraint, seed, sen_attr), "rb") as fp:
            influences = np.array(pickle.load(fp))
            influences = (influences - influences.min()) / (influences.max() - influences.min())

        pivot = int(train_dataset_length[dataset] * (k / 100.0))
        if method == 'naive_leave_k_out':
            fair_top = np.argpartition(influences, -pivot)[-pivot:]
               
            if option == 'intersect' or option == 'intersect_fine_tuning':
                with open("./influence_score/fair_only/{}_val_loss_influence_score_seed_{}_sen_attr_{}.txt".format(dataset, seed, sen_attr), "rb") as fp:
                    influences_val_loss = np.array(pickle.load(fp))
                            
                val_loss_top = np.argpartition(influences_val_loss, -pivot)[-pivot:]
                remove_idx = np.intersect1d(fair_top, val_loss_top)
            elif option == 'fair_with_val_loss':
                with open("./influence_score/fair_only/{}_val_loss_influence_score_seed_{}_sen_attr_{}.txt".format(dataset, seed, sen_attr), "rb") as fp:
                    # print("here")
                    influences_val_loss = np.array(pickle.load(fp))
                    influences_val_loss = (influences_val_loss - influences_val_loss.min()) / (influences_val_loss.max() - influences_val_loss.min())
                    
                    influences = alpha * influences + (1 - alpha) * influences_val_loss
                    remove_idx = np.argpartition(influences, -pivot)[-pivot:]
            else: remove_idx = fair_top
    

        elif method == 'naive_leave_bottom_k_out':
          
            fair_bottom = np.argpartition(influences, pivot)[:pivot]
                     
            if option == 'intersect' or option == 'intersect_fine_tuning':
                with open("./influence_score/fair_only/{}_{}_val_loss_influence_score_seed_{}_sen_attr_{}.txt".format(dataset, fairness_constraint, seed, sen_attr), "rb") as fp:
                    influences_val_loss = np.array(pickle.load(fp))
                val_loss_bottom = np.argpartition(influences_val_loss, pivot)[:pivot]
                remove_idx = np.intersect1d(fair_bottom, val_loss_bottom)
            elif option == 'fair_with_val_loss':
                with open("./influence_score/fair_only/{}_val_loss_influence_score_seed_{}_sen_attr_{}.txt".format(dataset, seed, sen_attr), "rb") as fp:
                    influences_val_loss = np.array(pickle.load(fp))
                    influences_val_loss = (influences_val_loss - influences_val_loss.min()) / (influences_val_loss.max() - influences_val_loss.min())

                    influences = alpha * influences + (1 - alpha) * influences_val_loss
                    remove_idx = np.argpartition(influences, pivot)[:pivot]
            else: remove_idx = fair_bottom
           
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

    #assert len(train_loader.dataset) == train_dataset_length[dataset]
    test_acc_arr = []
    test_fairness_metric_arr = []
    trng_acc_arr = []
    trng_fairness_metric_arr = []
    valid_acc_arr = []
    valid_fairness_metric_arr = []

    confu_mat_arr = []

    if dataset not in tabular_dataset:
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model = shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1, verbose=True)
    else:
        if dataset == 'adult': feature_size = 98
        elif dataset == 'compas': feature_size = 401
        elif dataset == 'bank': feature_size = 57
        elif dataset == 'retiring_adult': feature_size = 812
        elif dataset == 'retiring_adult_coverage': feature_size = 131

        if fine_tuning == 0:
            model = MLP(
                feature_size=feature_size,
                hidden_dim=50,
                num_classes=2,
                num_layer=2
            )
        elif fine_tuning == 1:
            model = torch.load("./model/{}/{}_MLP_target_{}_seed_{}_sen_attr_{}".format(option, dataset, target, seed, sen_attr))
        #optimizer = SGD(model.parameters(), lr=0.03, weight_decay=5e-4)
        optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
        #scheduler = ReduceLROnPlateau(optimizer, 'max', patience=10, verbose=True)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1, verbose=True)
    criterion = nn.CrossEntropyLoss(reduction='none')

    if method == 'naive':

        # scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=2)
        # scheduler = CosineAnnealingLR(optimizer, T_max=10, verbose=True)

        best_acc = 0.0
        for _epoch in tqdm(range(epoch)):
            trng_acc = 0.0
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

            #scheduler.step(test_acc)
            scheduler.step()
            if test_acc * 100 >= best_acc:
                print('Test Accuracy: {:.2f}, Model Save!'.format(test_acc * 100))
                # torch.save(model.state_dict(), './model/{}_resnet18_target_{}_seed_{}'.format(dataset, target, seed))
                if dataset not in tabular_dataset: torch.save(model, './model/{}/{}_{}_shufflenet_target_{}_seed_{}_sen_attr_{}'.format(option, dataset, fairness_constraint,  target, seed, sen_attr))
                else: torch.save(model, './model/{}/{}_MLP_target_{}_seed_{}_sen_attr_{}'.format(option, dataset, target, seed, sen_attr))

                best_acc = test_acc * 100

            # scheduler.step()

        log_arr = [trng_acc_arr, trng_fairness_metric_arr, valid_acc_arr, valid_fairness_metric_arr, test_acc_arr,
                   test_fairness_metric_arr]

        with open(log_epi + "_naive_log.txt".format(option, dataset, fairness_constraint, seed, sen_attr), "wb") as fp:
            pickle.dump(log_arr, fp)
          
        with open(log_epi + "_naive_confusion_matrix.txt".format(option, dataset, fairness_constraint, seed, sen_attr), "wb") as fp:
            pickle.dump(confu_mat_arr, fp)



    elif method == "naive_leave_k_out" or method == 'naive_leave_bottom_k_out':
        for _epoch in tqdm(range(epoch)):
            trng_acc = 0.0
            model.train()
            i = 0

            for z, _, _, t, tup in tqdm(train_loader):
                if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
               
                y_pred = model(z)
                if len(y_pred.shape) != 2:
                    y_pred = torch.unsqueeze(y_pred, 0)

                loss = torch.mean(criterion(y_pred, t))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                trng_acc += torch.sum(get_accuracy(y_pred, t, reduction='none')).item()

                i += 1
            
            if log_option == 'all':
                trng_acc /= len(train_loader.dataset)
                trng_acc_arr.append(trng_acc)

                confu_mat_train = compute_confusion_matrix(train_loader, model)
                trng_fairness_metric = calc_fairness_metric(args.constraint, confu_mat_train)
                trng_fairness_metric_arr.append(trng_fairness_metric)

                confu_mat_train = [confu_mat_train['0'], confu_mat_train['1']]
                print("Trng Acc {:.2f}, Trng fairness metric: {:.2f}".format(trng_acc * 100, trng_fairness_metric * 100))
            
            elif log_option == 'last':
                if _epoch == epoch - 1:
                    trng_acc /= len(train_loader.dataset)
                    trng_acc_arr.append(trng_acc)

                    confu_mat_train = compute_confusion_matrix(train_loader, model)
                    trng_fairness_metric = calc_fairness_metric(args.constraint, confu_mat_train)
                    trng_fairness_metric_arr.append(trng_fairness_metric)

                    confu_mat_train = [confu_mat_train['0'], confu_mat_train['1']]
                    print("Trng Acc {:.2f}, Trng fairness metric: {:.2f}".format(trng_acc * 100, trng_fairness_metric * 100))
                else:
                    trng_acc_arr.append(None)
                    trng_fairness_metric_arr.append(None)
                    confu_mat_train = [np.zeros((2,2)), np.zeros((2,2))]


            scheduler.step()

            if log_option == 'all' or (log_option == 'last' and _epoch == epoch - 1):
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
                test_fairness_metric = calc_fairness_metric(args.constraint, confu_mat_test)

                confu_mat_test = [confu_mat_test['0'], confu_mat_test['1']]
                print("Test Acc: {:.2f}, Test fairness metric: {:.2f}".format(test_acc * 100, test_fairness_metric * 100))
                test_acc_arr.append(test_acc.item())
                test_fairness_metric_arr.append(test_fairness_metric)

                valid_acc = 0.0
                if fairness_constraint == "eopp":
                    loss_arr = np.zeros(2)
                    group_size = np.zeros(2)
                elif fairness_constraint == "eo":
                    loss_arr = np.zeros((2, 2))

                with torch.no_grad():
                    for i, data in enumerate(valid_loader):
                        z, _, groups, t, _ = data
                        if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
                        y_pred = model(z)

                        # labels = t
                        # groups = groups.long()

                        # group_element = list(torch.unique(groups).numpy())
                        # if fairness_constraint == "eopp":
                        #     loss = nn.CrossEntropyLoss(reduction='none')(y_pred, t)

                            # for g in group_element:
                            #     group_mask = (groups == g)
                            #     label_mask = (labels == 1)

                            #     if torch.cuda.is_available():
                            #         group_mask = group_mask.cuda()
                            #         label_mask = label_mask.cuda()
                                
                            #     mask = torch.logical_and(group_mask, label_mask)
                            #     with torch.no_grad():
                            #         loss_arr[g] += torch.sum(loss[mask]).item()
                            #     group_size[g] += sum(mask).item()

                        valid_acc += torch.sum(get_accuracy(y_pred, t, reduction='none'))
                    valid_acc /= len(valid_loader.dataset)
                # loss_arr[0] /= group_size[0]
                # loss_arr[1] /= group_size[1]

                # print("valid violation: ", abs(loss_arr[0] - loss_arr[1]))

                confu_mat_valid = compute_confusion_matrix(valid_loader, model)
                valid_fairness_metric = calc_fairness_metric(args.constraint, confu_mat_valid)

                confu_mat_valid = [confu_mat_valid['0'], confu_mat_valid['1']]
                print("Valid Acc: {:.2f}, Valid fairness metric: {:.2f}".format(valid_acc * 100, valid_fairness_metric * 100))

                valid_acc_arr.append(valid_acc.item())
                valid_fairness_metric_arr.append(valid_fairness_metric)
            else:
                valid_acc_arr.append(None)
                valid_fairness_metric_arr.append(None)
                test_acc_arr.append(None)
                test_fairness_metric_arr.append(None)
                confu_mat_valid = np.zeros((2,2))
                confu_mat_test = np.zeros((2,2))

            confu_mat_arr.append([confu_mat_train, confu_mat_valid, confu_mat_test])

        acc_fair_log_arr = [trng_acc_arr, trng_fairness_metric_arr, valid_acc_arr, valid_fairness_metric_arr, test_acc_arr, test_fairness_metric_arr]

        with open(log_epi + "_acc_fair_log.txt".format(option, dataset, fairness_constraint, seed, k,sen_attr), "wb") as fp:
            pickle.dump(acc_fair_log_arr, fp)

        with open(log_epi + "_removed_data_info.txt".format(option, dataset, fairness_constraint, seed, k, sen_attr), "wb") as fp:
            pickle.dump(removed_data_log, fp)
    
        with open(log_epi + "_confusion_matrix.txt".format(option, dataset, fairness_constraint, seed, k, sen_attr), "wb") as fp:
            pickle.dump(confu_mat_arr, fp)

        # torch.save(model, f"./model/{option}/{dataset}_MLP_target_N")

if __name__ == '__main__':
    main()
