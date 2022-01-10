import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd
import itertools
import random
import config
from numpy.random import default_rng


def get_data(dataset, n_train, n_query, n_test):
    if dataset == "boston":
        x, y, pairs = get_boston_data()
    elif dataset == "styblinski_tang":
        x, y, pairs = get_styblinski_tang_data()
    elif dataset == "six_hump_camel":
        x, y, pairs = get_six_hump_camel_data()
    elif dataset == "forrester":
        x, y, pairs = get_forrester_data()
    elif dataset == "branin":
        x, y, pairs = get_branin_data(1)
    elif dataset == "levy":
        x, y, pairs = get_levy_data(1)
    elif dataset == "ackley":
        x, y, pairs = get_ackley_data()
    elif dataset == "triazines":
        x, y, pairs = get_triazines_data()
    elif dataset == "pyrimidine":
        x, y, pairs = get_pyrimidine_data()
    elif dataset == "machine":
        x, y, pairs = get_machine_data()
    else:
        raise NotImplementedError

    train_pairs = pairs[:n_train]
    query_pairs = pairs[n_train:n_train + n_query]
    test_pairs = pairs[n_train + n_query: n_train + n_query + n_test]

    x_duels_train = np.array(
        [[x[train_pairs[index][0]], x[train_pairs[index][1]]] for index in range(len(train_pairs))])
    pref_train = []
    for index in range(len(train_pairs)):
        pref_train.append(1) if y[train_pairs[index][0]] < y[train_pairs[index][1]] else pref_train.append(0)

    x_duels_query = np.array(
        [[x[query_pairs[index][0]], x[query_pairs[index][1]]] for index in range(len(query_pairs))])
    pref_query = []
    for index in range(len(query_pairs)):
        pref_query.append(1) if y[query_pairs[index][0]] < y[query_pairs[index][1]] else pref_query.append(0)

    x_duels_test = np.array([[x[test_pairs[index][0]], x[test_pairs[index][1]]] for index in range(len(test_pairs))])
    pref_test = []
    for index in range(len(test_pairs)):
        pref_test.append(1) if y[test_pairs[index][0]] < y[test_pairs[index][1]] else pref_test.append(0)

    train = {'x_duels': x_duels_train, 'pref': pref_train}
    query = {'x_duels': x_duels_query, 'pref': pref_query}
    test = {'x_duels': x_duels_test, 'pref': pref_test}

    return train, query, test


def forrester_function(x):
    return (6*x-2)**2 * np.sin(12*x-4)


def branin_function(x1, x2):
    a = 1
    b = 5.1 / (4*(np.pi**2))
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8*np.pi)
    y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1-t) * np.cos(x1) + s
    return y


def get_branin_data(seed):
    rng = default_rng(seed)
    x1 = rng.uniform(low=-5, high=10, size=1000)
    x2 = rng.uniform(low=0, high=15, size=1000)
    x = np.hstack([x1.reshape(-1, 1), x2.reshape(-1, 1)])
    y = branin_function(x1, x2)
    pairs = list(itertools.combinations(range(len(y)), 2))
    random.seed(config.seed)
    random.shuffle(pairs)
    return x, y, pairs


def levy_function(x):
    pi = np.pi

    x = 1 + (x-1)/4

    part1 = np.power(np.sin(pi * x[:, 0]), 2)

    part2 = np.sum(np.power(x[:, :-1]-1, 2) * (1 + 10 * np.power(np.sin(pi*x[:, :-1]+1), 2)), axis=1)

    part3 = np.power(x[:, -1]-1, 2) * (1 + np.power(np.sin(2 * pi * x[:, -1]), 2))

    y = part1 + part2 + part3
    return y


def get_levy_data(seed):
    rng = default_rng(32)
    x = rng.uniform(low=0, high=2, size=(1000, 10))
    y = levy_function(x)
    pairs = list(itertools.combinations(range(len(y)), 2))
    random.seed(config.seed)
    random.shuffle(pairs)

    return x, y, pairs


def styblinski_tang_function(x):
    y = 0.5 * np.sum(x**4 - 16*x**2 + 5*x, axis=1)

    return y


def six_hump_camel_function(x1, x2):
    y = (4 - 2.1 * (x1 ** 2) + (x1 ** 4) / 3) * (x1 ** 2) + x1 * x2 + (-4 + 4 * (x2 ** 2)) * (x2 ** 2)
    return y


def logistic_function(x):
    return 1 / (1+np.e**(-x))


def ackley_function(x):
    return -20 * np.exp(-0.2 * np.sqrt(1/20 * np.sum(x**2, axis=1))) - np.exp(1/20 * np.sum(np.cos(2*np.pi*x), axis=1)) + 20 + np.exp(1)


def get_ackley_data():
    rng = default_rng()
    x = rng.uniform(low=-32.768, high=32.768, size=(1000, 20))
    y = ackley_function(x)
    pairs = list(itertools.combinations(range(len(y)), 2))
    random.seed(config.seed)
    random.shuffle(pairs)
    print(min(y))
    return x, y, pairs


def get_forrester_data():
    x = np.linspace(0, 1, 100).reshape(-1, 1)
    y = forrester_function(x)
    pairs = list(itertools.permutations(range(len(x)), 2))
    random.seed(config.seed)
    random.shuffle(pairs)
    return x, y, pairs


def get_six_hump_camel_data():
    rng = default_rng()
    x1 = rng.uniform(low=-3, high=3, size=1000)
    x2 = rng.uniform(low=-2, high=2, size=1000)
    x = np.hstack([x1.reshape(-1, 1), x2.reshape(-1, 1)])
    y = six_hump_camel_function(x1, x2)
    pairs = list(itertools.combinations(range(len(y)), 2))
    random.seed(config.seed)
    random.shuffle(pairs)

    return x, y, pairs


def get_styblinski_tang_data():
    rng = default_rng()
    x = rng.uniform(low=-5, high=5, size=(1000, 20))
    y = styblinski_tang_function(x)
    pairs = list(itertools.combinations(range(len(y)), 2))
    random.seed(config.seed)
    random.shuffle(pairs)
    print(min(y))
    return x, y, pairs


def get_boston_data():
    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    boston['MEDV'] = boston_dataset.target
    boston = (boston - boston.mean()) / boston.std()
    pairs = list(itertools.combinations(range(len(boston)), 2))
    # random.seed(config.seed)
    random.shuffle(pairs)

    y = boston["MEDV"].to_numpy()
    x = boston.drop(columns=["MEDV"]).to_numpy()

    return x, y, pairs


def get_triazines_data():
    triazines = pd.read_csv('../triazines.csv')

    # triazines = (triazines - triazines.mean()) / triazines.std()

    pairs = list(itertools.combinations(range(len(triazines)), 2))
    random.seed(config.seed)
    random.shuffle(pairs)

    y = triazines["activity"].to_numpy()
    x = triazines.drop(columns=["activity"]).to_numpy()

    return x, y, pairs


def get_pyrimidine_data():
    pyrimidine = pd.read_csv('../pyrimidine.txt')

    # triazines = (triazines - triazines.mean()) / triazines.std()

    pairs = list(itertools.combinations(range(len(pyrimidine)), 2))
    # random.seed(config.seed)
    random.shuffle(pairs)

    y = pyrimidine["y"].to_numpy()
    x = pyrimidine.drop(columns=["y"]).to_numpy()

    return x, y, pairs


def get_machine_data():
    machine = pd.read_csv('../machine.txt')

    machine = (machine - machine.mean()) / machine.std()

    pairs = list(itertools.combinations(range(len(machine)), 2))
    # random.seed(config.seed)
    random.shuffle(pairs)

    y = machine["y"].to_numpy()
    x = machine.drop(columns=["y"]).to_numpy()

    return x, y, pairs


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
        diff = x2 - x1
        diff = diff.squeeze(1)
        indic = torch.pow(-1, pref)
        sigmoid = nn.Sigmoid()

        loss = indic * sigmoid(diff)
        return torch.sum(loss)


def plot_acc_trend(nn_list, acc_nn_std, gp_list, acc_gp_std, fig_name):
    nb = [i for i in range(len(nn_list[0]))]
    plt.plot(nb, gp_list[0], c="orange", label="GP-R")
    plt.scatter(nb, gp_list[0], c="orange", marker='.', s=80)
    plt.plot(nb, gp_list[1], c="green", label="GP-LC")
    plt.scatter(nb, gp_list[1], c="green", marker='.', s=80)
    plt.plot(nb, gp_list[2], c="yellow", label="GP-BALD")
    plt.scatter(nb, gp_list[2], c="yellow", marker='.', s=80)

    plt.plot(nb, nn_list[0], c="blue", label="PBNN-R")
    plt.scatter(nb, nn_list[0], c="blue", marker=',', s=16)
    plt.plot(nb, nn_list[1], c="grey", label="PBNN-LC")
    plt.scatter(nb, nn_list[1], c="grey", marker=',', s=16)
    plt.plot(nb, nn_list[2], c="red", label="PBNN-BALD")
    plt.scatter(nb, nn_list[2], c="red", marker=',', s=16)
    plt.gca().fill_between(nb,
                           nn_list[0]-acc_nn_std[0]/20,
                           nn_list[0]+acc_nn_std[0]/20, color="blue", alpha=0.1)
    plt.gca().fill_between(nb,
                           nn_list[1]-acc_nn_std[1]/20,
                           nn_list[1]+acc_nn_std[1]/20, color="grey", alpha=0.1)
    plt.gca().fill_between(nb,
                           nn_list[2] - acc_nn_std[2] / 20,
                           nn_list[2] + acc_nn_std[2] / 20, color="red", alpha=0.1)

    plt.gca().fill_between(nb,
                           gp_list[0] - acc_gp_std[0] / 20,
                           gp_list[0] + acc_gp_std[0] / 20, color="orange", alpha=0.1)
    plt.gca().fill_between(nb,
                           gp_list[1] - acc_gp_std[1] / 20,
                           gp_list[1] + acc_gp_std[1] / 20, color="green", alpha=0.1)
    plt.gca().fill_between(nb,
                           gp_list[2] - acc_gp_std[2] / 20,
                           gp_list[2] + acc_gp_std[2] / 20, color="yellow", alpha=0.1)
    plt.legend(loc=7)
    plt.savefig(fig_name)
    plt.close()
    # plt.show()


def plot_function_shape(x, y, pred):
    plt.plot(x, pred)
    plt.plot(x, y, c="red", label="True")
    plt.scatter(x[np.argmin(pred)], np.min(pred), marker="*", c="black")
    plt.scatter(x[np.argmin(y)], np.min(y), marker="^", c="blue")
    plt.show()


if __name__ == "__main__":
    print(levy_function(np.array([[1,1,1]])))
    get_levy_data(1)