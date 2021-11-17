import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np


class PrefNet(nn.Module):
    def __init__(self, n_input):
        super(PrefNet, self).__init__()
        self.fc1 = nn.Linear(n_input, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(100, 30)
        # self.bn2 = nn.BatchNorm1d(30)
        self.fc3 = nn.Linear(30, 1)

        torch.nn.init.normal_(self.fc1.weight)
        torch.nn.init.normal_(self.fc2.weight)
        torch.nn.init.normal_(self.fc3.weight)

    def forward_once(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.fc3(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# variance propagation
def variance_product_rnd_vars(mean1, mean2, var1, var2):
    return mean1 ** 2 * var2 + mean2 ** 2 * var1 + var1 * var2


class UDropout(nn.Module):
    def __init__(self, rate, initial_noise=False):
        super(UDropout, self).__init__()
        self.initial_noise = initial_noise
        self.rate = rate
        self.dropout = nn.Dropout(rate)

    def _call_diag_cov(self, mean, var):
        if self.initial_noise:
            out = mean ** 2 * self.rate / (1 - self.rate)
        else:
            new_mean = 1 - self.rate
            new_var = self.rate * (1 - self.rate)
            out = variance_product_rnd_vars(mean, new_mean, var, new_var) / (1 - self.rate) ** 2
        return out

    def forward(self, inp):
        mean, var = inp
        return self.dropout(mean), self._call_diag_cov(mean, var)


class ULinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ULinear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def _call_diag_cov(self, var):
        return F.linear(var, self.linear.weight ** 2)

    def forward(self, inp):
        mean, var = inp
        return self.linear(mean), self._call_diag_cov(var)


class UReLU(nn.Module):
    def __init__(self):
        super(UReLU, self).__init__()
        self.eps = 1e-8

    def _call_diag_cov(self, mean, var):
        std = torch.sqrt(var + self.eps)
        exp = mean / (np.sqrt(2.0) * std)
        erf_exp = torch.erf(exp)
        exp_exp2 = torch.exp(-1 * exp ** 2)
        term1 = 0.5 * (var + mean ** 2) * (erf_exp + 1)
        term2 = mean * std / (np.sqrt(2 * math.pi)) * exp_exp2
        term3 = mean / 2 * (1 + erf_exp)
        term4 = np.sqrt(1 / 2 / math.pi) * std * exp_exp2
        return F.relu(term1 + term2 - (term3 + term4) ** 2)

    def forward(self, inp):
        mean, var = inp
        return F.relu(mean), self._call_diag_cov(mean, var)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.fc1 = ULinear(1, 100)
        self.fc2 = ULinear(100, 30)
        self.fc3 = ULinear(30, 1)

        self.dropout = UDropout(0.1)
        self.relu = UReLU()

    def forward(self, inp):

        mean, var = self.fc1(inp)
        mean, var = self.relu((mean, var))
        mean, var = self.dropout((mean, var))
        mean, var = self.fc2((mean, var))
        mean, var = self.relu((mean, var))
        mean, var = self.dropout((mean, var))
        mean, var = self.fc3((mean, var))

        return mean, var
