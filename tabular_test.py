from data_handler.dataloader_factory import DataloaderFactory


kwargs = {
    'name': 'adult',
    'batch_size': 128,
    'seed': 0,
    'num_workers': 0
}
num_classes, num_groups, train_loader, valid_loader, test_loader = DataloaderFactory.get_dataloader(**kwargs)

for X, _,_,_,_ in train_loader:
    print(X.shape)