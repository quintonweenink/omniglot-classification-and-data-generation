import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn


class CNN(torch.nn.Module):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # def __init__(self, out1, out2):
    #     super().__init__()
    #     self.layer1 = torch.nn.Sequential(
    #         torch.nn.Conv2d(1, out1, kernel_size=3, stride=1, padding=1),
    #         torch.nn.ReLU(),
    #         torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    #     )
    #     self.layer2 = torch.nn.Sequential(
    #         torch.nn.Conv2d(out1, out2, kernel_size=3, stride=1, padding=1),
    #         torch.nn.ReLU(),
    #         torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    #     )
    #     # print(self.output_size(105, 2, 2, 0))
    #     self.fc1 = torch.nn.Linear(21632, 1000)
    #     self.bat_norm = torch.nn.BatchNorm1d(1000)
    #     self.fc2 = torch.nn.Linear(1000, 964)
    #
    # def forward(self, x):
    #     # print("size1:", x.size())
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     # print("size2:", x.size())
    #     x = x.view(-1, 32 * x.size(2) * x.size(3))
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     x = self.bat_norm(x)
    #     x = self.fc2(x)
    #     # print(x.size())
    #     x = F.softmax(x, dim=1)
    #     # print(x, torch.min(x), torch.max(x))
    #
    #     return x
    def __init__(self, out1=None, out2=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # self.adapt_pool1 = nn.AdaptiveMaxPool2d((6, 49, 49))
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.adapt_pool2 = nn.AdaptiveMaxPool2d((16, 21, 21))
        self.fc1 = nn.Linear(16 * 23 * 23, 4232)
        self.bat_norm = torch.nn.BatchNorm1d(4232)
        self.fc2 = nn.Linear(4232, 964)
        # self.fc3 = nn.Linear(3528, 964)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        # print("After Conv1:", x.size())
        x = F.relu(x)
        x = self.pool(x)
        # print("After pool:", x.size())
        x = self.conv2(x)
        # print("After Conv2:", x.size())
        x = F.relu(x)
        x = self.pool(x)
        # print("After pool:", x.size())
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bat_norm(x)
        print(x.size())
        # x = F.dropout(x, p=0.5)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def loss_and_optimiser(self, lr):
        return torch.nn.CrossEntropyLoss(), optim.Adam(self.parameters(), lr=lr)

    @staticmethod
    def output_size(in_size, kernel_size, stride, padding):
        output = int((in_size - kernel_size + 2 * padding) / stride) + 1
        return output
