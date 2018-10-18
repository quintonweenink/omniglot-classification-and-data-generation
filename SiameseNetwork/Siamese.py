import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn


class Siamese(torch.nn.Module):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    def __init__(self, out1=None, out2=None):
        super().__init__()
        self.l1_distance = lambda tensor0, tensor1: torch.abs(tensor0 - tensor1)
        self.cnn = nn.Sequential(nn.Conv2d(1, 6, 5),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2, 2),
                                 nn.Conv2d(6, 16, 5),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2, 2)
                                 )
        self.full = nn.Sequential(nn.Linear(16 * 23 * 23, 4232),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.5),
                                  torch.nn.BatchNorm1d(4232),
                                  nn.Linear(4232, 1),
                                  nn.Sigmoid()
                                  )

    def forward_once(self, x):
        output = self.cnn(x)
        # print(output.size())
        output = output.view(-1, output.size(1) * output.size(2) * output.size(3))
        return output

    def forward(self, x):
        # print(x[0].size(), x[1].size())
        # print("1")
        output1 = self.forward_once(x[0])
        # print("2")
        output2 = self.forward_once(x[1])

        output = self.l1_distance(output1, output2)

        output = self.full(output)

        return output

    def loss_and_optimiser(self, lr):
        return nn.BCELoss(), optim.Adam(self.parameters(), lr=lr)

    @staticmethod
    def output_size(in_size, kernel_size, stride, padding):
        output = int((in_size - kernel_size + 2 * padding) / stride) + 1
        return output
