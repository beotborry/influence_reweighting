import torch
import pickle
import numpy as np
from tqdm import tqdm

def calc_fairness_metric(constraint, confu_mat, num_groups=2, num_classes=2):
    if constraint == 'eopp':
        '''
        Compute EO disparity
        '''
        group0_tn, group0_fp, group0_fn, group0_tp = confu_mat['0'].ravel()
        group1_tn, group1_fp, group1_fn, group1_tp = confu_mat['1'].ravel()

        pivot = (group0_tp + group1_tp) / (group0_fn + group0_tp + group1_fn + group1_tp)
        group0_tpr = group0_tp / (group0_fn + group0_tp)
        group1_tpr = group1_tp / (group1_fn + group1_tp)

        return max(abs(group0_tpr - pivot), abs(group1_tpr - pivot)) # from fairbatch paper
        #return abs(group0_tp / (group0_fn + group0_tp) - group1_tp / (group1_fn + group1_tp))

    elif constraint == 'eo':
        '''
        Compute ED disparity 
        '''

        # group0_tn, group0_fp, group0_fn, group0_tp = confu_mat['0'].ravel()
        # group1_tn, group1_fp, group1_fn, group1_tp = confu_mat['1'].ravel()
        
        # pivot_1 = (group0_tp + group1_tp) / (group0_fn + group0_tp + group1_fn + group1_tp)
        # group0_tpr = group0_tp / (group0_fn + group0_tp)
        # group1_tpr = group1_tp / (group1_fn + group1_tp)

        # EO_Y_1 = max(abs(group0_tpr - pivot_1), abs(group1_tpr - pivot_1))

        # pivot_0 = (group0_fp + group1_fp) / (group0_tn + group0_fp + group1_tn + group1_fp)
        # group0_fpr = (group0_fp) / (group0_tn + group0_fp)
        # group1_fpr = (group1_fp) / (group1_tn + group1_fp)

        # EO_Y_0 = max(abs(group0_fpr - pivot_0), abs(group1_fpr - pivot_0))

        # return max(EO_Y_0, EO_Y_1)
        
        group0_tn, group0_fp, group0_fn, group0_tp = confu_mat['0'].ravel()
        group1_tn, group1_fp, group1_fn, group1_tp = confu_mat['1'].ravel()
        
        group0_tpr = group0_tp / (group0_fn + group0_tp)
        group1_tpr = group1_tp / (group1_fn + group1_tp)

        group0_tnr = group0_tn / (group0_tn + group0_fp)
        group1_tnr = group1_tn / (group1_tn + group1_fp)

        return (abs(group0_tpr - group1_tpr) + abs(group0_tnr - group1_tnr)) / 2

    elif constraint == 'dp':
        pass


def grad_V(constraint, dataloader, model, _dataset, _seed, _sen_attr, main_option, save=False):
    params = [p for p in model.parameters() if p.requires_grad]
    if torch.cuda.is_available(): model = model.cuda()
    if constraint == 'eopp':
        losses = [0.0, 0.0]
        group_size = [0, 0]
        result = []

        for i, data in tqdm(enumerate(dataloader)):
            inputs, _, groups, targets, _ = data

            labels = targets
            groups = groups.long()

            if torch.cuda.is_available(): 
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)

            # for influence score w.r.t val loss
            if i == 0:
                grad_val_loss = list(grad(torch.sum(loss), params, retain_graph=True))
            elif i > 0:
                curr = list(grad(torch.sum(loss), params, retain_graph=True))
                for idx in range(len(grad_val_loss)):
                    grad_val_loss[idx] += curr[idx]

            group_element = list(torch.unique(groups).numpy())
            for g in group_element:

                group_mask = (groups == g)
                label_mask = (labels == 1)
                if torch.cuda.is_available():
                    group_mask = group_mask.cuda()
                    label_mask = label_mask.cuda()

                mask = torch.logical_and(group_mask, label_mask)

                with torch.no_grad():
                    losses[g] += torch.sum(loss[mask]).item()

                if group_size[g] == 0 and g == 0: grad_0 = list(grad(torch.sum(loss[mask]), params, retain_graph=True))
                elif group_size[g] == 0 and g == 1: grad_1 = list(grad(torch.sum(loss[mask]), params, retain_graph=True))
                
                if group_size[g] != 0 and g == 0: 
                    curr = list(grad(torch.sum(loss[mask]), params, retain_graph=True))
                    for idx in range(len(grad_0)):
                        grad_0[idx] += curr[idx]
                elif group_size[g] != 0 and g == 1:
                    curr = list(grad(torch.sum(loss[mask]), params, retain_graph=True))
                    for idx in range(len(grad_1)):
                        grad_1[idx] += curr[idx]

                group_size[g] += sum(mask).item()

        print(group_size)
        losses[0] /= group_size[0]
        losses[1] /= group_size[1]

        if losses[0] > losses[1]:
            for elem in zip(grad_0, grad_1):
                result.append(elem[0] / group_size[0] - elem[1] / group_size[1])
        else:
            for elem in zip(grad_0, grad_1):
                result.append(elem[1] / group_size[1] - elem[0] / group_size[0])

        for elem in grad_val_loss:
            elem /= len(dataloader.dataset)
                
        if save == True:
            with open("./influence_score/{}/{}_{}_val_loss_gradV_seed_{}_sen_attr_{}.txt".format(main_option, _dataset, constraint, _seed, _sen_attr), "wb") as fp:
                pickle.dump(grad_val_loss, fp)

            with open("./influence_score/{}/{}_{}_gradV_seed_{}_sen_attr_{}.txt".format(main_option, _dataset, constraint,  _seed, _sen_attr), "wb") as fp:
                pickle.dump(result, fp)
        else:
            return result

    elif constraint == 'eo':
        losses = np.zeros((2, 2))
        group_size = np.zeros((2, 2))
        result = []

        for i, data in tqdm(enumerate(dataloader)):
            inputs, _, groups, targets, _ = data
            labels = targets
            groups = groups.long()

            if torch.cuda.is_available(): inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)

            group_element = list(torch.unique(groups).numpy())
            for g in group_element:
                for l in [0, 1]:
                    group_mask = (groups == g)
                    label_mask = (labels == l)
                    if torch.cuda.is_available():
                        group_mask = group_mask.cuda()
                        label_mask = label_mask.cuda()
                    
                    mask = torch.logical_and(group_mask, label_mask)

                    with torch.no_grad():
                        losses[g, l] += torch.sum(loss[mask]).item()

                    if group_size[g, l] == 0 and g == 0 and l == 0:
                        grad_00 = list(grad(torch.sum(loss[mask]), params, retain_graph=True))
                    elif group_size[g, l] == 0 and g == 0 and l == 1:
                        grad_01 = list(grad(torch.sum(loss[mask]), params, retain_graph=True))
                    elif group_size[g, l] == 0 and g == 1 and l == 0:
                        grad_10 = list(grad(torch.sum(loss[mask]), params, retain_graph=True))
                    elif group_size[g, l] == 0 and g == 1 and l == 1:
                        grad_11 = list(grad(torch.sum(loss[mask]), params, retain_graph=True))
                    
                    if group_size[g, l] != 0:
                        curr = list(grad(torch.sum(loss[mask]), params, retain_graph=True))
                        if g == 0 and l == 0:
                            for idx in range(len(grad_00)):
                                grad_00[idx] += curr[idx]
                        elif g == 0 and l == 1:
                            for idx in range(len(grad_01)):
                                grad_01[idx] += curr[idx]
                        elif g == 1 and l == 0:
                            for idx in range(len(grad_10)):
                                grad_10[idx] += curr[idx]
                        elif g == 1 and l == 1:
                            for idx in range(len(grad_11)):
                                grad_11[idx] += curr[idx]

                    group_size[g, l] += sum(mask).item()

        print(group_size)
        for g in group_element:
            for l in [0, 1]:
                losses[g, l] /= group_size[g, l]

        if losses[0, 0] > losses[1, 0] and losses[0, 1] > losses[1, 1]:
            for elem in zip(grad_00, grad_10, grad_01, grad_11):
                result.append(((elem[0] / group_size[0, 0] - elem[1] / group_size[1, 0]) + (elem[2] / group_size[0, 1] - elem[3] / group_size[1, 1])) / 2)
        elif losses[0, 0] < losses[1, 0] and losses[0, 1] > losses[1, 1]:
            for elem in zip(grad_00, grad_10, grad_01, grad_11):
                result.append(((elem[1] / group_size[1, 0] - elem[0] / group_size[0, 0]) + (elem[2] / group_size[0, 1] - elem[3] / group_size[1, 1])) / 2)
        elif losses[0, 0] > losses[1, 0] and losses[0, 1] < losses[1, 1]:
            for elem in zip(grad_00, grad_10, grad_01, grad_11):
                result.append(((elem[0] / group_size[0, 0] - elem[1] / group_size[1, 0]) + (elem[3] / group_size[1, 1] - elem[2] / group_size[0, 1])) / 2)
        elif losses[0, 0] < losses[1, 0] and losses[0, 1] < losses[1, 1]:
            for elem in zip(grad_00, grad_10, grad_01, grad_11):
                result.append(((elem[1] / group_size[1, 0] - elem[0] / group_size[0, 0]) + (elem[3] / group_size[1, 1] - elem[2] / group_size[0, 1])) / 2)
        # if abs(losses[0, 0] - losses[1, 0]) > abs(losses[0, 1] - losses[1, 1]):
        #     if losses[0, 0] > losses[1, 0]:
        #         for elem in zip(grad_00, grad_10):
        #             result.append(elem[0] / group_size[0,0] - elem[1] / group_size[1, 0])
        #     elif losses[0, 0] < losses[1, 0]:
        #         for elem in zip(grad_00, grad_10):
        #             result.append(elem[1] / group_size[1, 0] - elem[0] / group_size[0, 0])
        # elif abs(losses[0, 0] - losses[1, 0]) < abs(losses[0, 1] - losses[1, 1]):
        #     if losses[0, 1] > losses[1, 1]:
        #         for elem in zip(grad_01, grad_11):
        #             result.append(elem[0] / group_size[0, 1] - elem[1] / group_size[1,1])
        #     elif losses[0,1] < losses[1,1]:
        #         for elem in zip(grad_01, grad_11):
        #             result.append(elem[1]/group_size[1,1] - elem[0]/group_size[0,1])
        if save == True:
            with open("./influence_score/{}/{}_{}_gradV_seed_{}_sen_attr_{}.txt".format(main_option, _dataset, constraint, _seed, _sen_attr), "wb") as fp:
                pickle.dump(result, fp)
        else:
            return result