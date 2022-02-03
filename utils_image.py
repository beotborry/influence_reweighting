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
        group0_tn, group0_fp, group0_fn, group0_tp = confu_mat['0'].ravel()
        group1_tn, group1_fp, group1_fn, group1_tp = confu_mat['1'].ravel()

        pivot = (group0_tp + group1_tp) / (group0_fn + group0_tp + group1_fn + group1_tp)
        group0_tpr = group0_tp / (group0_fn + group0_tp)
        group1_tpr = group1_tp / (group1_fn + group1_tp)

        return max(abs(group0_tpr - pivot), abs(group1_tpr - pivot))
        #return abs(group0_tp / (group0_fn + group0_tp) - group1_tp / (group1_fn + group1_tp))

    elif constraint == 'eo':
        '''
        Compute DEO_A
        '''
        result = 0.0
        eval_eo_list = np.zeros((num_groups, num_classes))
        eval_data_count = np.zeros((num_groups, num_classes))

        for g in range(num_groups):
            for l in range(num_classes):
                eval_data_count[g, l] = sum(confu_mat['g'][l,:]).float()
                eval_eo_list[g, l] = confu_mat['g'][l,l]
        
        eval_eo_prob = eval_eo_list / eval_data_count

        for l in range(num_classes):
            result += max(eval_eo_prob[:, l]) - min(eval_eo_prob[:, l])
        
        return result / num_classes

    elif constraint == 'dp':
        pass

