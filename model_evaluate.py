from collections import defaultdict
from email.policy import default
from pandas import to_pickle
import torch
import argparse
from data_handler import DataloaderFactory
from utils import set_seed
import numpy as np
import pickle

from utils_image import compute_confusion_matrix

def get_args():
    parser = argparse.ArgumentParser(description="influence score based weighting")
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--dataset', required=True, default='', choices=['adult', 'compas', 'bank', 'credit', 'celeba', 'utkface', 'retiring_adult', 'retiring_adult_coverage'])
    parser.add_argument('--seed', required=True, default=0, type=int)
    parser.add_argument('--sen_attr', required=True, type=str)
    parser.add_argument('--constraint', required=True, type=str)
    
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    dataset = args.dataset
    seed = args.seed
    sen_attr = args.sen_attr
    gpu = args.gpu
    constraint = args.constraint

    torch.cuda.set_device(f"cuda:{gpu}")
    set_seed(gpu)


    model = torch.load("./model/fair_only/{}_MLP_target_None_seed_{}_sen_attr_{}".format(dataset, seed, sen_attr))

    num_classes, num_groups, train_loader, valid_loader, test_loader = DataloaderFactory.get_dataloader(name=dataset,
                                                                                            batch_size=128,
                                                                                            seed=seed,
                                                                                            num_workers=0,
                                                                                            target=None,
                                                                                            influence_scores=[],
                                                                                            sen_attr=sen_attr)

    with open(f"./influence_score/fair_only/{dataset}_{constraint}_influence_score_seed_{seed}_sen_attr_{sen_attr}.txt", "rb") as fp:
        influence_score = pickle.load(fp)

    k = 0.2
    num_remove = int(len(train_loader.dataset) * k * 0.01)
    top_k_idx = np.argpartition(influence_score, -num_remove)[-num_remove:]

    removed_data_confu_mat = defaultdict(lambda: np.zeros((2, 2)))

    with torch.no_grad():
        for idx in top_k_idx:
            feature, _, group, label, _ = train_loader.dataset[idx]
            feature = torch.tensor(feature).unsqueeze(0).cuda()
            print(model(feature))
            y_pred = model(feature).argmax().item()

            removed_data_confu_mat[str(group)][label, y_pred] += 1

    print(removed_data_confu_mat.values())


    total_confu_mat = compute_confusion_matrix(valid_loader, model)

    
    print(total_confu_mat['0'])
    print(total_confu_mat['1'])

    total_confu_mat = compute_confusion_matrix(test_loader, model)

    
    print(total_confu_mat['0'])
    print(total_confu_mat['1'])

    # loss_group = np.zeros(2)
    # cnt_group = np.zeros(2)

    # with torch.no_grad():
    #     for i, data in enumerate(valid_loader):
    #         z, _, groups, t, _ = data

    #         labels = t
    #         groups = groups.long()
    #         if torch.cuda.is_available():
    #             z, t = z.cuda(), t.cuda()
    #         y_pred = model(z)

    #         loss = torch.nn.CrossEntropyLoss(reduction='none')(y_pred, t)

    #         for g in [0, 1]:
    #             group_mask = (groups == g)
    #             label_mask = (labels == 1)
    #             mask = torch.logical_and(group_mask, label_mask)

    #             cnt_group[g] += len(mask)
    #             loss_group[g] += loss[mask].sum().item()
            
            
    # print("vio: ", abs(loss_group[0] / cnt_group[0] - loss_group[1] / cnt_group[1]))

                

if __name__ == "__main__":
    main()