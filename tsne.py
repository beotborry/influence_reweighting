from sklearn.manifold import TSNE
from data_handler.dataset_factory import DatasetFactory
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

data = DatasetFactory.get_dataset("adult", split="train", influence_scores=[])
train_loader = DataLoader(data, batch_size=len(data))

data = next(iter(train_loader))[0].numpy()
labels = data[:, -1]


n_components = 2
model = TSNE(n_components=n_components)
result = model.fit_transform(data)
print(result)

plt.scatter(result[:, 0], result[:, 1], c=labels)
plt.show()
