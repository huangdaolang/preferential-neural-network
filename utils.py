import numpy as np
import torch.nn as nn
import torch


def forrester_function(x):
    return (6*x-2)**2 * np.sin(12*x-4)


def logistic_function(x):
    return 1 / (1+np.e**(-x))


# preference loss function for neural network
class PrefLoss_Forrester(nn.Module):
    def __init__(self):
        super(PrefLoss_Forrester, self).__init__()

    def forward(self, x1, x2, pref):
        diff = x2 - x1
        diff = diff.squeeze(1)
        indic = torch.pow(-1, pref)
        sigmoid = nn.Sigmoid()

        loss = indic * sigmoid(diff)
        return torch.sum(loss)

