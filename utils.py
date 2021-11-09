import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


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


def plot_acc_trend(nb, gp_list, nn_list):
    plt.plot(nb, gp_list, c="red", label="gp")
    plt.scatter(nb, gp_list, c="red", marker='.', s=120)
    plt.plot(nb, nn_list, c="blue", label="nn")
    plt.scatter(nb, nn_list, c="blue", marker=',')
    plt.legend()
    plt.show()
    plt.savefig("comp.png")


def plot_function_shape(x, y, pred):
    plt.plot(x, pred)
    plt.plot(x, y, c="red", label="True")
    plt.scatter(x[np.argmin(pred)], np.min(pred), marker="*", c="black")
    plt.scatter(x[np.argmin(y)], np.min(y), marker="^", c="blue")
    plt.show()
