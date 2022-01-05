import torch
from torch.random import seed
from data_handler.dataloader_factory import DataloaderFactory
from utils import get_accuracy
from utils_image import compute_confusion_matrix, calc_fairness_metric
from argument import get_args

def main():
    args = get_args()
    GPU_NUM = args.gpu
    dataset = args.dataset
    target = args.target
    fairness_constraint = args.constraint
    seed = args.seed

    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(device)
    print(device)

    model = torch.load("./model/{}_resnet18_target_{}".format(dataset, target))
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
    print('Test Fairness Metric: {:.2f}%'.format(calc_fairness_metric(fairness_constraint, confu_mat) * 100))

if __name__ == '__main__':
    main()
