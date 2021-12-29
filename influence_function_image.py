import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pickle
from torch.autograd import grad
import time

def grad_z(z, t, model, gpu=-1):
    if z.dim() == 3: z = torch.unsqueeze(z, 0)
    if t.dim() != 1: t = t.view(1)

    model.eval()
    if torch.cuda.is_available():
        z, t, model = z.cuda(), t.cuda(), model.cuda()
    y = model(z)
    if y.dim() == 1: y = y.view(1,2)

    loss = torch.nn.CrossEntropyLoss()(y, t)

    params = [p for p in model.parameters() if p.requires_grad]
    return list(grad(loss, params, create_graph=True))

def get_eopp_idx(dataset, load=False):
    if load == False:
        g0y1 = []
        g1y1 = []

        for i, data in tqdm(enumerate(dataset)):
            _, _, group, target, _  = data
            if group == 0 and target == 1: g0y1.append(i)
            elif group == 1 and target == 1: g1y1.append(i)

        ret = []
        ret.append(np.array(g0y1))
        ret.append(np.array(g1y1))
        with open("./celeba_eopp_idx.txt", "wb") as fp:
            pickle.dump(ret, fp)
    elif load == True:
        with open("./celeba_eopp_idx.txt", "rb") as fp:
            ret = pickle.load(fp)
        
    return ret

def get_eo_idx(dataset):
    g0y0 = []
    g0y1 = []
    g1y0 = []
    g1y1 = []

    for i, data in enumerate(dataset):
        _, _, group, target, _ = data
        if group == 0 and target == 0: g0y0.append(i)
        elif group == 0 and target == 1: g0y1.append(i)
        elif group == 1 and target == 0: g1y0.append(i)
        elif group == 1 and target == 1: g1y1.append(i)

    ret = []
    ret.append(np.array(g0y0))
    ret.append(np.array(g0y1))
    ret.append(np.array(g1y0))
    ret.append(np.array(g1y1))

    return ret


def grad_V_tmp(constraint, dataset, model):
    model.eval()

    params = [p for p in model.parameters() if p.requires_grad]

    if constraint == 'eopp':
        losses = [0.0, 0.0]
        eopp_idx = get_eopp_idx(dataset, load=True)
        grads = [None, None]

        with torch.no_grad():
            for i, idx_arr in enumerate(eopp_idx):
                for idx in tqdm(idx_arr):
                    X, _, _, target, _ = dataset[idx]
                    if torch.cuda.is_available(): X, target, model = X.cuda(), target.cuda(), model.cuda()
                    X = torch.unsqueeze(X, 0)
                    target = torch.unsqueeze(target, 0)
                    losses[i] += nn.CrossEntropyLoss()(model(X), target)
        
        print("calc loss done")

        for i, idx_arr in enumerate(eopp_idx):
            for idx in tqdm(idx_arr):
                X, _, _, target, _ = dataset[idx]
                if torch.cuda.is_available(): X, target, model = X.cuda(), target.cuda(), model.cuda()
                X = torch.unsqueeze(X, 0)
                target = torch.unsqueeze(target, 0)
                loss = nn.CrossEntropyLoss()(model(X), target)
                _grad = grad(loss, params, create_graph=True)
                
                if grads[i] == None: grads[i] = _grad
                else: grads[i] += grad
            grads[i] /= len(idx_arr)

        if losses[0] >= losses[1]:
            return list(grads[0] - grads[1])
        else: return list(grads[1] - grads[0])
        #violation = abs(losses[0] - losses[1])
        return list(grad(violation, params, create_graph=True))



    elif constraint == 'eo':
        pass
    elif constraint == 'dp':
        pass


def grad_V(constraint, dataloader, model, save=False):
    params = [p for p in model.parameters() if p.requires_grad]
    if torch.cuda.is_available(): model = model.cuda()
    if constraint == 'eopp':
        grad_0 = []
        grad_1 = []
        for i, data in enumerate(dataloader):
            inputs, _, groups, targets, _ = data
            labels = targets
            groups = groups.long()

            if torch.cuda.is_available(): 
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)

            group_element = list(torch.unique(groups).numpy())

            losses = torch.tensor([0.0, 0.0])
            if torch.cuda.is_available(): losses = losses.cuda()

            for g in group_element:
               # mask = np.where(np.logical_and(groups == i, labels == 1))
                group_mask = groups == g
                label_mask = targets == 1
                #print(torch.logical_and(group_mask,label_mask))
                mask = torch.logical_and(group_mask, label_mask)
                losses[g] += torch.sum(loss[mask])
                #print(list(grad(loss_sum, params, create_graph=True)))
            
        losses[0] /= 83342
        losses[1] /= 43446

        loss_diff = abs(losses[0] - losses[1])
        #print(list(grad(loss_diff, params, create_graph=True)))
           # _grad = []
           # for elem in grad(loss_sum, params, create_graph=True):
           #     _grad.append(elem.cpu())
           # #with torch.no_grad():
           # #    _grad = list(grad(loss_sum, params, create_graph=True))

           # with torch.no_grad():
           #     if i == 0 and g == 0: grad_0 = _grad
           #     elif i == 0 and g == 1: grad_1 = _grad
           #     elif g == 0 : grad_0 += [i + j for i in grad_0 for j in _grad]
           #     elif g == 1 : grad_1 += [i + j for i in grad_1 for j in _grad]
        if save == True:
            with open("celeba_gradV_seed_100.txt", "wb") as fp:
                pickle.dump(list(grad(loss_diff, params, create_graph=True)), fp)
        else:
            return list(grad(loss_diff, params, create_graph=True))

def s_test(model, dataloader, random_sampler, constraint, weights, recursion_depth=100, damp=0.01, scale=500.0, load_gradV=False, save=False):
    model.eval()

    if load_gradV == False:
         v = grad_V(constraint, dataloader, model, save=False)
    else:
        #grad_V(constraint, dataloader, model, save=True)
        with open("celeba_gradV_seed_100.txt", "rb") as fp:
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
        start = time.time()
        hv = hvp(loss[i], params, h_estimate)

        start = time.time()
        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]

    if save == True:
        with open("celeba_s_test_seed_100.txt", "wb") as fp:
            pickle.dump(h_estimate, fp)

    return h_estimate

def avg_s_test(model, dataloader, random_sampler, constraint, weights, r, recursion_depth=100, damp=0.01, scale=500.0, save=True):

    all = s_test(model, dataloader, random_sampler, constraint, weights, recursion_depth, damp, scale, load_gradV=False, save=False)

    for i in tqdm(range(1, r)):
        cur = s_test(model, dataloader, random_sampler, constraint, weights, recursion_depth, damp, scale, load_gradV=True, save=False)
        all = [a + c for a, c in zip(all, cur)]

    all = [a / r for a in all]
    if save == True:
        with open("celeba_s_test_avg_seed_100.txt", "wb") as fp:
            pickle.dump(all, fp)
    return all


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

def calc_influence_dataset(model, dataloader, random_sampler, constraint, weights, recursion_depth=5000, damp=0.01, scale=25.0, load_s_test=True):
    if load_s_test == True:
        with open("celeba_s_test_avg_seed_100.txt", "rb") as fp:
            s_test_vec = pickle.load(fp)
    else: s_test_vec = avg_s_test(model, dataloader, random_sampler, constraint, weights, recursion_depth, damp, scale, save=True)

    influences = np.zeros(len(dataloader.dataset))
    torch.cuda.synchornize()
    for i, data in enumerate(dataloader):
        X, _, _, t, tup = data
        for X_elem, t_elem, idx in zip(X, t, tup[0]):
            influences[idx] = calc_influence(X, t, s_test_vec, model, len(dataloader.dataset)).cpu()
   
    return influences




