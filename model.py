import torch.nn as nn
import torch.nn.functional as F
import torch


class PrefNet_Forrester(nn.Module):
    def __init__(self):
        super(PrefNet_Forrester, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        # self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(100, 30)
        # self.bn2 = nn.BatchNorm1d(30)
        self.fc3 = nn.Linear(30, 1)
        torch.nn.init.normal_(self.fc1.weight)
        torch.nn.init.normal_(self.fc2.weight)
        torch.nn.init.normal_(self.fc3.weight)

    def forward_once(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        # x = self.dropout(x)
        x = F.relu(x)

        x = self.fc2(x)
        # x = self.bn2(x)
        # x = self.dropout(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

