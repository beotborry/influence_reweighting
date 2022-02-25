from tokenize import group
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import defaultdict

def compute_confusion_matrix(dataloader, model, num_classes=2):
    model.eval()
    if torch.cuda.is_available(): model = model.cuda()
    
    confu_mat = defaultdict(lambda: np.zeros((num_classes, num_classes)))

    predict_mat = {}
    output_set = torch.tensor([])
    group_set = torch.tensor([], dtype=torch.long)
    target_set = torch.tensor([], dtype=torch.long)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, _, groups, targets, _ = data
            labels = targets
            groups = groups.long()

            if torch.cuda.is_available(): inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            
            group_set = torch.cat((group_set, groups))
            target_set = torch.cat((target_set, targets))
            if len(outputs.shape) != 2:
                outputs = torch.unsqueeze(outputs, 0)
            output_set = torch.cat((output_set, outputs.cpu()))

            pred = torch.argmax(outputs, 1)
            group_element = list(torch.unique(groups).numpy())
            for i in group_element:
                mask = groups == i
                if len(labels[mask]) != 0:
                    confu_mat[str(i)] += confusion_matrix(
                        labels[mask].cpu().numpy(), pred[mask].cpu().numpy(),
                        labels=[i for i in range(num_classes)])

    predict_mat['group_set'] = group_set.numpy()
    predict_mat['target_set'] = target_set.numpy()
    predict_mat['output_set'] = output_set.numpy()

    return confu_mat

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
