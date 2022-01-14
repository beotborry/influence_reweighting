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

def grad_V(constraint, dataloader, model, _dataset, _seed, save=False):
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

            group_element = list(torch.unique(groups).numpy())
            for g in group_element:

                group_mask = (groups == g).cuda()
                label_mask = (labels == 1).cuda()

                mask = torch.logical_and(group_mask, label_mask)

                with torch.no_grad():
                    losses[g] += torch.sum(loss[mask]).item()

                if group_size[g] == 0 and g == 0: grad_0 = list(grad(torch.sum(loss[mask]), params, retain_graph=True))
                elif group_size[g] == 0 and g == 1: grad_1 = list(grad(torch.sum(loss[mask]), params, retain_graph=True))
                
                if group_size[g] != 0 and g == 0: 
                    curr = list(grad(torch.sum(loss[mask]), params, retain_graph=True))
                    for i in range(len(grad_0)):
                        grad_0[i] += curr[i]
                elif group_size[g] != 0 and g == 1:
                    curr = list(grad(torch.sum(loss[mask]), params, retain_graph=True))
                    for i in range(len(grad_1)):
                        grad_1[i] += curr[i]

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

                
        if save == True:
            with open("./influence_score/{}_gradV_seed_{}.txt".format(_dataset, _seed), "wb") as fp:
                pickle.dump(result, fp)
        else:
            return result

def s_test(model, dataloader, random_sampler, constraint, weights, _dataset, _seed, recursion_depth=100, damp=0.01, scale=500.0, load_gradV=False, save=False):
    model.eval()

    if load_gradV == False:
         v = grad_V(constraint, dataloader, model, _dataset, _seed, save=False)
    else:
        #grad_V(constraint, dataloader, model, save=True)
        with open("./influence_score/{}_gradV_seed_{}.txt".format(_dataset, _seed), "rb") as fp:
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
        with open("./influence_score/{}_s_test_seed_{}.txt".format(_dataset, _seed), "wb") as fp:
            pickle.dump(h_estimate, fp)

    return h_estimate

def avg_s_test(model, dataloader, random_sampler, constraint, weights, r, _dataset, _seed, recursion_depth=100, damp=0.01, scale=500.0, save=True):

    all = s_test(model, dataloader, random_sampler, constraint, weights, _dataset, _seed, recursion_depth, damp, scale, load_gradV=True, save=False)

    for i in tqdm(range(1, r)):
        cur = s_test(model, dataloader, random_sampler, constraint, weights, _dataset, _seed, recursion_depth, damp, scale, load_gradV=True, save=False)
        all = [a + c for a, c in zip(all, cur)]

    all = [a / r for a in all]
    if save == True:
        with open("./influence_score/{}_s_test_avg_seed_{}.txt".format(_dataset, _seed), "wb") as fp:
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

def calc_influence_dataset(model, dataloader, s_test_dataloader, random_sampler, constraint, weights, _dataset, _seed, recursion_depth=5000, r=1, damp=0.01, scale=25.0, load_s_test=True):
    if load_s_test == True:
        with open("./influence_score/{}_s_test_avg_seed_{}.txt".format(_dataset, _seed), "rb") as fp:
            s_test_vec = pickle.load(fp)
    else: s_test_vec = avg_s_test(model, s_test_dataloader, random_sampler, constraint, weights, r, _dataset, _seed, recursion_depth, damp, scale, save=True)

    influences = np.zeros(len(dataloader.dataset))
    torch.cuda.synchronize()
    for i, data in tqdm(enumerate(dataloader)):
        X, _, _, t, tup = data
        for X_elem, t_elem, idx in zip(X, t, tup[0]):
            #print(X_elem, idx)
            influences[idx] = calc_influence(X_elem, t_elem, s_test_vec, model, len(dataloader.dataset)).cpu()
   
    return influences
