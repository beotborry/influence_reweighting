class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, transform=None, split='Train', target='Attractive', seed=0, skew_ratio=1., labelwise=False):

        if name == 'celeba':
            from data_handler.celeba import CelebA
            root='./data/'
            return CelebA(root=root, split=split, transform=transform, target_attr=target, labelwise=labelwise)
        
        elif name == 'utkface':
            from data_handler.utkface import UTKFaceDataset
            root='./data/utkface'
            return UTKFaceDataset(root=root, split=split, transform=transform, labelwise=labelwise)


