import torch
from torch.random import seed
from data_handler.dataloader_factory import DataloaderFactory
from utils import get_accuracy
from utils_image import compute_confusion_matrix, calc_fairness_metric
import argparse
from utils import set_seed
from mlp import MLP
from torch.utils.data import DataLoader

def model_evaluate(dataset, target, fairness_constraint, seed, gpu):
    set_seed(seed)

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(device)
    print(device)

    if dataset == "adult":
        if dataset == "adult":
            from adult_dataloader import get_data
            from adult_dataloader import Adult as CustomDataset

        _X_train, _y_train, X_test, y_test, _protected_train, protected_test = get_data()
        
        pivot = int(len(_X_train) * 0.8)
        X_train = _X_train[:pivot]
        y_train = _y_train[:pivot]
        protected_train = [_protected_train[0][:pivot], _protected_train[1][:pivot]]
        
        X_valid = _X_train[pivot:]
        y_valid = _y_train[pivot:]
        protected_valid = [_protected_train[0][pivot:], _protected_train[1][pivot:]]

        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_valid = torch.FloatTensor(X_valid)
        y_valid = torch.LongTensor(y_valid)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)

        train_dataset = CustomDataset(X_train, y_train, protected_train)
        valid_dataset = CustomDataset(X_valid, y_valid, protected_valid)
        test_dataset = CustomDataset(X_test, y_test, protected_test)

        train_loader = DataLoader(train_dataset, batch_size = 128, shuffle= False, num_workers = 0)
        valid_loader = DataLoader(valid_dataset, batch_size = 128, shuffle=False, num_workers = 0)
        test_loader = DataLoader(test_dataset, batch_size = 128, shuffle=False, num_workers = 0)

        model = torch.load("./model/{}_MLP_target_{}_seed_{}".format(dataset, target, seed))
    else:
        model = torch.load("./model/{}_resnet18_target_{}_seed_{}".format(dataset, target, seed))
        #model = torch.load("./model/celeba_resnet18_target_young")

        num_classes, num_groups, train_loader, test_loader, valid_loader = DataloaderFactory.get_dataloader(dataset, img_size=128, batch_size=128, seed=seed, num_workers=4, target=target)

    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            z, _, _, t, _ = data
            if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
            y_pred = model(z)
            test_acc += torch.sum(get_accuracy(y_pred, t, reduction='none')).item()
        
        test_acc /= len(test_loader.dataset)
    
    print('Test Acc: {:.2f}%'.format(test_acc * 100))
    
    confu_mat = compute_confusion_matrix(test_loader, model)
    #print(confu_mat['0'].ravel())
    #print(confu_mat['1'].ravel())
    test_fairness_metric = calc_fairness_metric(fairness_constraint, confu_mat) * 100
    print('Test Fairness Metric: {:.2f}%'.format(test_fairness_metric))

    valid_acc = 0.0
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            z, _, _, t, _ = data
            if torch.cuda.is_available(): z, t, model = z.cuda(), t.cuda(), model.cuda()
            y_pred = model(z)
            valid_acc += torch.sum(get_accuracy(y_pred, t, reduction='none')).item()
        valid_acc /= len(valid_loader.dataset)

    print("Valid Acc: {:.2f}%".format(valid_acc * 100))

    confu_mat = compute_confusion_matrix(valid_loader, model)

    valid_fairness_metric = calc_fairness_metric(fairness_constraint, confu_mat) * 100
    print("Valid Fairness Metric: {:.2f}%".format(valid_fairness_metric))

    return test_acc * 100, test_fairness_metric

if __name__ == '__main__':
    model_evaluate()
