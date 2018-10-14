import torchvision
from torchvision.datasets import Omniglot


omniglot = Omniglot(root="./data", download=True)
omniglot.download()

image = omniglot.__getitem__(300)[0]
print(image)



import matplotlib.pyplot as plt

plt.hist()

import torch.nn as nn
import torch.nn.functional as F

trainset = torchvision.datasets.Omniglot(root='./data',
                                        download=True, transform=transform)


torchvision.datasets.ImageFolder
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

trainset.__getitem__(0)

criterion = nn.CrossEntropyLoss()
criterion([], [])

F.tanh

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

import os

os.rmdir


net = Net()

print("== DONE ==")



