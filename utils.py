import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd
import itertools
import random
import config


def forrester_function(x):
    return (6*x-2)**2 * np.sin(12*x-4)


def logistic_function(x):
    return 1 / (1+np.e**(-x))


def get_forrester_data():
    x = np.linspace(0, 1, 100)
    nn_x_test = torch.from_numpy(x).reshape(-1, 1)
    y = forrester_function(x)
    pairs = list(itertools.permutations(range(len(x)), 2))
    random.shuffle(pairs)
    return x, y, nn_x_test, pairs


def get_boston_data(n_train, n_query, n_test):
    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    boston['MEDV'] = boston_dataset.target
    boston = (boston - boston.mean()) / boston.std()
    pairs = list(itertools.combinations(range(len(boston)), 2))
    random.seed(2021)
    random.shuffle(pairs)

    y = boston["MEDV"].to_numpy()
    x = boston.drop(columns=["MEDV"]).to_numpy()

    train_pairs = pairs[:n_train]
    query_pairs = pairs[n_train:n_train+n_query]
    test_pairs = pairs[n_train+n_query: n_train+n_query+n_test]

    x_duels_train = np.array([[x[train_pairs[index][0]], x[train_pairs[index][1]]] for index in range(len(train_pairs))])
    pref_train = []
    for index in range(len(train_pairs)):
        pref_train.append(1) if y[train_pairs[index][0]] > y[train_pairs[index][1]] else pref_train.append(0)

    x_duels_query = np.array([[x[query_pairs[index][0]], x[query_pairs[index][1]]] for index in range(len(query_pairs))])
    pref_query = []
    for index in range(len(query_pairs)):
        pref_query.append(1) if y[query_pairs[index][0]] > y[query_pairs[index][1]] else pref_query.append(0)

    x_duels_test = np.array([[x[test_pairs[index][0]], x[test_pairs[index][1]]] for index in range(len(test_pairs))])
    pref_test = []
    for index in range(len(test_pairs)):
        pref_test.append(1) if y[test_pairs[index][0]] > y[test_pairs[index][1]] else pref_test.append(0)

    train = {'x_duels': x_duels_train, 'pref': pref_train}
    query = {'x_duels': x_duels_query, 'pref': pref_query}
    test = {'x_duels': x_duels_test, 'pref': pref_test}

    return train, query, test


def get_gp_input(x, y, dim):
    gp_x = x.reshape(-1, dim)
    # gp_x_test = x.reshape(-1, dim)
    gp_m = []
    for a in range(len(x)):
        for b in range(len(x)):
            if y[a] > y[b]:
                gp_m.append([a, b])
            elif y[a] < y[b]:
                gp_m.append([b, a])
    return gp_x, gp_m


# preference loss function for neural network
class PrefLoss_Forrester(nn.Module):
    def __init__(self):
        super(PrefLoss_Forrester, self).__init__()

    def forward(self, x1, x2, pref):
        diff = x1 - x2
        diff = diff.squeeze(1)
        indic = torch.pow(-1, pref)
        sigmoid = nn.Sigmoid()

        loss = indic * sigmoid(diff)
        return torch.sum(loss)


def plot_acc_trend(nn_list, gp_list, fig_name):
    nb = [i for i in range(len(nn_list[0]))]
    plt.plot(nb, gp_list[0], c="red", label="gp_random")
    plt.scatter(nb, gp_list[0], c="red", marker='.', s=120)
    plt.plot(nb, gp_list[1], c="green", label="gp_uncertainty")
    plt.scatter(nb, gp_list[1], c="green", marker='.', s=120)
    plt.plot(nb, nn_list[0], c="blue", label="nn_random")
    plt.scatter(nb, nn_list[0], c="blue", marker=',')
    plt.plot(nb, nn_list[1], c="black", label="nn_uncertainty")
    plt.scatter(nb, nn_list[1], c="black", marker=',')
    plt.legend()
    plt.savefig(fig_name)
    plt.close()
    # plt.show()


def plot_function_shape(x, y, pred):
    plt.plot(x, pred)
    plt.plot(x, y, c="red", label="True")
    plt.scatter(x[np.argmin(pred)], np.min(pred), marker="*", c="black")
    plt.scatter(x[np.argmin(y)], np.min(y), marker="^", c="blue")
    plt.show()
