from multiprocessing import reduction
from tokenize import group
from warnings import resetwarnings
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pickle
from torch.autograd import grad

def grad_z(z, t, model, gpu=-1):
    if z.dim() == 1: z = torch.unsqueeze(z, 0)
    if z.dim() == 3: z = torch.unsqueeze(z, 0)
    if t.dim() != 1: t = t.view(1)

    model.eval()
    if torch.cuda.is_available():
        z, t, model = z.cuda(), t.cuda(), model.cuda()
    y = model(z)
    if y.dim() == 1: y = y.view(1,2)

    loss = torch.nn.CrossEntropyLoss()(y, t)

    params = [p for p in model.parameters() if p.requires_grad]
    return list(grad(loss, params, retain_graph=True))

def grad_V(constraint, dataloader, model, _dataset, _seed, _sen_attr, main_option, save=False, split=False):
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

        print("violation: ", abs(losses[0] - losses[1]))
        # print(0 / 0)

        if losses[0] > losses[1]:
            for elem in zip(grad_0, grad_1):
                result.append(elem[0] / group_size[0] - elem[1] / group_size[1])
        else:
            for elem in zip(grad_0, grad_1):
                result.append(elem[1] / group_size[1] - elem[0] / group_size[0])

        for elem in grad_val_loss:
            elem /= len(dataloader.dataset)
                
        if save == True and split == False:
            with open("./influence_score/{}/{}_val_loss_gradV_seed_{}_sen_attr_{}.txt".format(main_option, _dataset, _seed, _sen_attr), "wb") as fp:
                pickle.dump(grad_val_loss, fp)

            with open("./influence_score/{}/{}_{}_gradV_seed_{}_sen_attr_{}.txt".format(main_option, _dataset, constraint,  _seed, _sen_attr), "wb") as fp:
                pickle.dump(result, fp)

        elif save == True and split == True:
            with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_gradV_seed_{_seed}_sen_attr_{_sen_attr}_group0.txt", "wb") as fp:
                for i in range(len(grad_0)):
                    grad_0[i] /= group_size[0]
                pickle.dump(grad_0, fp)
            with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_gradV_seed_{_seed}_sen_attr_{_sen_attr}_group1.txt", "wb") as fp:
                for i in range(len(grad_1)):
                    grad_1[i] /= group_size[1]
                pickle.dump(grad_1, fp)

            with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_loss_info_seed_{_seed}_sen_attr_{_sen_attr}.txt", "wb") as fp:
                pickle.dump(losses, fp)
            
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
        print("violation: ", (abs(losses[0, 0] - losses[1, 0]) + abs(losses[0, 1] - losses[1, 1]) / 2))

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
        if save == True and split == False:
            with open("./influence_score/{}/{}_{}_gradV_seed_{}_sen_attr_{}.txt".format(main_option, _dataset, constraint, _seed, _sen_attr), "wb") as fp:
                pickle.dump(result, fp)
        
        elif save == True and split == True:
            
            with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_loss_info_seed_{_seed}_sen_attr_{_sen_attr}.txt", "wb") as fp:
                pickle.dump(losses, fp)
            
            with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_gradV_seed_{_seed}_sen_attr_{_sen_attr}_group0_label0.txt", "wb") as fp:
                for i in range(len(grad_00)):
                    grad_00[i] /= group_size[0][0]
                pickle.dump(grad_00, fp)
            
            with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_gradV_seed_{_seed}_sen_attr_{_sen_attr}_group0_label1.txt", "wb") as fp:
                for i in range(len(grad_01)):
                    grad_01[i] /= group_size[0][1]
                pickle.dump(grad_01, fp)

            with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_gradV_seed_{_seed}_sen_attr_{_sen_attr}_group1_label0.txt", "wb") as fp:
                for i in range(len(grad_10)):
                    grad_10[i] /= group_size[1][0]
                pickle.dump(grad_10, fp)

            with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_gradV_seed_{_seed}_sen_attr_{_sen_attr}_group1_label1.txt", "wb") as fp:
                for i in range(len(grad_11)):
                    grad_11[i] /= group_size[1][1]
                pickle.dump(grad_11, fp)



def s_test(model, dataloader, random_sampler, constraint, weights, _dataset, _seed, _sen_attr, main_option, option='fair',recursion_depth=100, damp=0.01, scale=500.0, load_gradV=False, save=False, split=False):
    model.eval()
    if split == False:
        if load_gradV == False:
            v = grad_V(constraint, dataloader, model, _dataset, _seed, save=False)
        else:
            if option == 'fair':
                with open("./influence_score/{}/{}_{}_gradV_seed_{}_sen_attr_{}.txt".format(main_option, _dataset, constraint, _seed, _sen_attr), "rb") as fp:
                    v = pickle.load(fp)
            elif option == 'val_loss':
                with open("./influence_score/{}/{}_val_loss_gradV_seed_{}_sen_attr_{}.txt".format(main_option, _dataset, _seed, _sen_attr), "rb") as fp:
                    v = pickle.load(fp)

        h_estimate = v.copy()
        
        params = [p for p in model.parameters() if p.requires_grad]

        for data in random_sampler:
            X, _, _, t, tup = data
            idx = tup[0]

            if torch.cuda.is_available():
                X, t, model, weights  = X.cuda(), t.cuda(), model.cuda(), weights.cuda()
            y = model(X)
            loss = weights[idx] * torch.nn.CrossEntropyLoss(reduction='none')(y, t)
            break

        for i in tqdm(range(recursion_depth)):
            hv = hvp(loss[i], params, h_estimate)

            with torch.no_grad():
                h_estimate = [
                    _v + (1 - damp) * _h_e - _hv / scale
                    for _v, _h_e, _hv in zip(v, h_estimate, hv)]

        if save == True:
            if option == 'fair':
                with open("./influence_score/{}_{}_s_test_seed_{}_sen_attr_{}.txt".format(_dataset, constraint, _seed, _sen_attr), "wb") as fp:
                    pickle.dump(h_estimate, fp)
            elif option == 'val_loss':
                with open("./influence_score/{}_val_loss_s_test_seed_{}_sen_attr_{}.txt".format(_dataset, _seed, _sen_attr), "wb") as fp:
                    pickle.dump(h_estimate, fp)
        return h_estimate

    elif split == True:

        if constraint == "eopp":
            with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_gradV_seed_{_seed}_sen_attr_{_sen_attr}_group0.txt", "rb") as fp:
                group0_gradV = pickle.load(fp)
            with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_gradV_seed_{_seed}_sen_attr_{_sen_attr}_group1.txt", "rb") as fp:
                group1_gradV = pickle.load(fp)

            s_test_arr = []    
            for g in range(2):
                if g == 0: v = group0_gradV
                elif g == 1: v = group1_gradV

                h_estimate = v.copy()

                params = [p for p in model.parameters() if p.requires_grad]

                for data in random_sampler:
                    X, _, _, t, tup = data
                    idx = tup[0]

                    if torch.cuda.is_available():
                        X, t, model, weights = X.cuda(), t.cuda(), model.cuda(), weights.cuda()

                    y = model(X)
                    loss = weights[idx] * torch.nn.CrossEntropyLoss(reduction='none')(y, t)
                    break

                for i in tqdm(range(recursion_depth)):
                    hv = hvp(loss[i], params, h_estimate)

                    with torch.no_grad():
                        h_estimate = [
                            _v + (1 - damp) * _h_e - _hv / scale for _v, _h_e, _hv in zip(v, h_estimate, hv)
                        ]
                
                s_test_arr.append(h_estimate)
            return s_test_arr
        elif constraint == "eo":

            with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_gradV_seed_{_seed}_sen_attr_{_sen_attr}_group0_label0.txt", "rb") as fp:
                group00_gradV = pickle.load(fp)
            with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_gradV_seed_{_seed}_sen_attr_{_sen_attr}_group0_label1.txt", "rb") as fp:
                group01_gradV = pickle.load(fp)
            with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_gradV_seed_{_seed}_sen_attr_{_sen_attr}_group1_label0.txt", "rb") as fp:
                group10_gradV = pickle.load(fp)
            with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_gradV_seed_{_seed}_sen_attr_{_sen_attr}_group1_label1.txt", "rb") as fp:
                group11_gradV = pickle.load(fp)

            s_test_arr = []    
            for g in range(2):
                for l in range(2):
                    if g == 0 and l == 0: v = group00_gradV
                    elif g == 0 and l == 1: v = group01_gradV
                    elif g == 1 and l == 0: v = group10_gradV
                    elif g == 1 and l == 1: v = group11_gradV

                    h_estimate = v.copy()

                    params = [p for p in model.parameters() if p.requires_grad]

                    for data in random_sampler:
                        X, _, _, t, tup = data
                        idx = tup[0]

                        if torch.cuda.is_available():
                            X, t, model, weights = X.cuda(), t.cuda(), model.cuda(), weights.cuda()

                        y = model(X)
                        loss = weights[idx] * torch.nn.CrossEntropyLoss(reduction='none')(y, t)
                        break

                    for i in tqdm(range(recursion_depth)):
                        hv = hvp(loss[i], params, h_estimate)

                        with torch.no_grad():
                            h_estimate = [
                                _v + (1 - damp) * _h_e - _hv / scale for _v, _h_e, _hv in zip(v, h_estimate, hv)
                            ]
                    
                    s_test_arr.append(h_estimate)
            return s_test_arr
def avg_s_test(model, dataloader, random_sampler, constraint, weights, r, _dataset, _seed, _sen_attr, main_option, option='fair', recursion_depth=100, damp=0.01, scale=500.0, save=True, split=False):

    if split == False:
        all = s_test(model, dataloader, random_sampler, constraint, weights, _dataset, _seed, _sen_attr, main_option, option, recursion_depth, damp, scale, load_gradV=True, save=False, split=split)

        for i in tqdm(range(1, r)):
            cur = s_test(model, dataloader, random_sampler, constraint, weights, _dataset, _seed, _sen_attr, main_option, option, recursion_depth, damp, scale, load_gradV=True, save=False, split=split)
            all = [a + c for a, c in zip(all, cur)]

        all = [a / r for a in all]
        if save == True:
            if option == 'fair':
                with open("./influence_score/{}/{}_{}_s_test_avg_seed_{}_sen_attr_{}.txt".format(main_option, _dataset, constraint, _seed, _sen_attr), "wb") as fp:
                    pickle.dump(all, fp)
            elif option == 'val_loss':
                with open("./influence_score/{}/{}_val_loss_s_test_avg_seed_{}_sen_attr_{}.txt".format(main_option, _dataset, _seed, _sen_attr), "wb") as fp:
                    pickle.dump(all, fp)
        return all
    
    elif split == True:
        s_test_arr = s_test(model, dataloader, random_sampler, constraint, weights, _dataset, _seed, _sen_attr, main_option, option, recursion_depth, damp, scale, load_gradV=True, save=False, split=split)
        
        for i in tqdm(range(1, r)):
            cur_arr = s_test(model, dataloader, random_sampler, constraint, weights, _dataset, _seed, _sen_attr, main_option, option, recursion_depth, damp, scale, load_gradV=True, save=False, split=split)
            
            if constraint == "eopp":
                for g in range(2):
                    s_test_arr[g] = [a + c for a, c in zip(s_test_arr[g], cur_arr[g])]
            elif constraint == "eo":
                for g in range(2):
                    for l in range(2):
                        s_test_arr[2 * g + l] = [a + c for a, c in zip(s_test_arr[2*g + l], cur_arr[2*g + l])]
            
        if constraint == "eopp":
            for g in range(2):
                s_test_arr[g] = [a / r for a in s_test_arr[g]]
        elif constraint == "eo":
            for g in range(2):
                for l in range(2):
                    s_test_arr[2*g + l] = [a / r for a in s_test_arr[2*g+l]]
        
        if save == True:
            if constraint == "eopp":
                with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_s_test_avg_seed_{_seed}_sen_attr_{_sen_attr}_group0.txt", "wb") as fp:
                    pickle.dump(s_test_arr[0], fp)
                with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_s_test_avg_seed_{_seed}_sen_attr_{_sen_attr}_group1.txt", "wb") as fp:
                    pickle.dump(s_test_arr[1], fp)
            elif constraint == "eo":
                with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_s_test_avg_seed_{_seed}_sen_attr_{_sen_attr}_group0_label0.txt", "wb") as fp:
                    pickle.dump(s_test_arr[0], fp)
                with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_s_test_avg_seed_{_seed}_sen_attr_{_sen_attr}_group0_label1.txt", "wb") as fp:
                    pickle.dump(s_test_arr[1], fp)
                with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_s_test_avg_seed_{_seed}_sen_attr_{_sen_attr}_group1_label0.txt", "wb") as fp:
                    pickle.dump(s_test_arr[2], fp)
                with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_s_test_avg_seed_{_seed}_sen_attr_{_sen_attr}_group1_label1.txt", "wb") as fp:
                    pickle.dump(s_test_arr[3], fp)               
            
        return s_test_arr

def hvp(y, w, v):
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    first_grads = grad(y, w, retain_graph = True, create_graph=True)

    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem.detach())
    
    return grad(elemwise_products, w, retain_graph=True)

def calc_influence(z, t, s_test, model, dataset_size):
    grad_z_vec = grad_z(z, t, model)
    influence = -sum([
        torch.sum(k*j).data for k, j in zip(grad_z_vec, s_test)]) / dataset_size
    
    return influence

def calc_influence_dataset(model, dataloader, s_test_dataloader, random_sampler, constraint, weights, _dataset, _seed, _sen_attr, main_option, option='fair', recursion_depth=5000, r=1, damp=0.01, scale=25.0, load_s_test=True, split = False):
    
    if split == False: 
        if load_s_test == True:
            if option == 'fair':
                with open("./influence_score/{}/{}_{}_s_test_avg_seed_{}_sen_attr_{}.txt".format(main_option, _dataset, constraint, _seed, _sen_attr), "rb") as fp:
                    s_test_vec = pickle.load(fp)
            elif option == 'val_loss':
                with open("./influence_score/{}/{}_val_loss_s_test_avg_seed_{}_sen_attr_{}.txt".format(main_option, _dataset, _seed, _sen_attr), "rb") as fp:
                    s_test_vec = pickle.load(fp)

        else: s_test_vec = avg_s_test(model, s_test_dataloader, random_sampler, constraint, weights, r, _dataset, _seed, _sen_attr, recursion_depth, damp, scale, save=True)

        influences = np.zeros(len(dataloader.dataset))
        torch.cuda.synchronize()
        for i, data in tqdm(enumerate(dataloader)):
            X, _, _, t, tup = data
            for X_elem, t_elem, idx in zip(X, t, tup[0]):
                #print(X_elem, idx)
                influences[idx] = calc_influence(X_elem, t_elem, s_test_vec, model, len(dataloader.dataset)).cpu()
    
        return influences
    
    elif split == True:
        if load_s_test == True:

            if constraint == "eopp":
                with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_s_test_avg_seed_{_seed}_sen_attr_{_sen_attr}_group0.txt", "rb") as fp:
                    s_test_vec_group0 = pickle.load(fp)
                with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_s_test_avg_seed_{_seed}_sen_attr_{_sen_attr}_group1.txt", "rb") as fp:
                    s_test_vec_group1 = pickle.load(fp)
                
                for g in range(2):
                    if g == 0: s_test_vec = s_test_vec_group0
                    elif g == 1: s_test_vec = s_test_vec_group1

                    influences = np.zeros(len(dataloader.dataset))
                    torch.cuda.synchronize()
                    for i, data in tqdm(enumerate(dataloader)):
                        X, _, _, t, tup = data
                        for X_elem, t_elem, idx in zip(X, t, tup[0]):
                            influences[idx] = calc_influence(X_elem, t_elem, s_test_vec, model, len(dataloader.dataset)).cpu()
                    
                    with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_influence_score_seed_{_seed}_sen_attr_{_sen_attr}_group{g}.txt", "wb") as fp:
                        pickle.dump(influences, fp)
            elif constraint == "eo":
                with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_s_test_avg_seed_{_seed}_sen_attr_{_sen_attr}_group0_label0.txt", "rb") as fp:
                    s_test_vec_group0_label0 = pickle.load(fp)
                with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_s_test_avg_seed_{_seed}_sen_attr_{_sen_attr}_group0_label1.txt", "rb") as fp:
                    s_test_vec_group0_label1 = pickle.load(fp)
                with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_s_test_avg_seed_{_seed}_sen_attr_{_sen_attr}_group1_label0.txt", "rb") as fp:
                    s_test_vec_group1_label0 = pickle.load(fp)
                with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_s_test_avg_seed_{_seed}_sen_attr_{_sen_attr}_group1_label1.txt", "rb") as fp:
                    s_test_vec_group1_label1 = pickle.load(fp)

                for g in range(2):
                    for l in range(2):
                        if g == 0 and l == 0: s_test_vec = s_test_vec_group0_label0
                        elif g == 0 and l == 1: s_test_vec = s_test_vec_group0_label1
                        elif g == 1 and l == 0: s_test_vec = s_test_vec_group1_label0
                        elif g == 1 and l == 1: s_test_vec = s_test_vec_group1_label1

                        influences = np.zeros(len(dataloader.dataset))
                        torch.cuda.synchronize()
                        for i, data in tqdm(enumerate(dataloader)):
                            X, _, _, t, tup = data
                            for X_elem, t_elem, idx in zip(X, t, tup[0]):
                                influences[idx] = calc_influence(X_elem, t_elem, s_test_vec, model, len(dataloader.dataset)).cpu()
                        
                        with open(f"./influence_score/{main_option}/{_dataset}_{constraint}_influence_score_seed_{_seed}_sen_attr_{_sen_attr}_group{g}_label{l}.txt", "wb") as fp:
                            pickle.dump(influences, fp)
            
        else: raise SystemError("no load s_test")
            
