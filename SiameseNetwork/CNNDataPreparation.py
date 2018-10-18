import numpy as np  # linear algebra
import torchvision
import Dataset.Omniglot as datasets
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import RandomlyApply as RA


class DataPrep(object):
    def __init__(self):
        transform = trans.Compose([trans.ToTensor(), trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # Apply at a random probability a random number of randomly chosen transformations
        transform_two = trans.Compose([trans.RandomApply([
                                                        trans.RandomRotation(degrees=45),
                                                        trans.RandomAffine(degrees=0, translate=(0, 0.2),
                                                                           scale=(0.3, 1.2), shear=(0, 45)),
                                                        trans.RandomHorizontalFlip()
                                                        ]),
                                        trans.ToTensor(),
                                        trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_set = torchvision.datasets.Omniglot(root='./omniglotdata', background=True, download=False,
                                           transform=transform_two)

        # dataset = torchvision.datasets.ImageFolder('omniglotdata/omniglot-py/', transform=transform_two)

        self.val_set = torchvision.datasets.Omniglot(root='./omniglotdata', background=True, download=False,
                                         transform=transform)
        self.test_set = torchvision.datasets.Omniglot(root='./omniglotdata', background=False, download=False,
                                          transform=transform)

        self.batch_size = 50
        # train_split = 70
        # print(len(self.train_set), end=' ')
        # print(len(self.test_set), str(len(self.test_set) + len(self.train_set)))
        # self.num_train_samples = len(self.train_set)
        # self.num_val_samples = self.num_train_samples*0.2

        # self.train_sampler = SubsetRandomSampler(list(np.arange(self.num_train_samples - self.num_val_samples,
        #                                                         dtype=np.int64)))
        # self.val_sampler = SubsetRandomSampler(list(np.arange(self.num_train_samples - self.num_val_samples,
        #                                                       self.num_train_samples, dtype=np.int64)))

        self.train_loader = DataLoader(self.train_set, batch_size=50, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(self.val_set, shuffle=True, batch_size=50, num_workers=2)
        self.test_loader = DataLoader(self.test_set, shuffle=True, batch_size=50, num_workers=2)

        # print(len(self.train_loader), len(self.test_loader))
        # print(len(self.train_loader.dataset), len(self.test_loader.dataset))
