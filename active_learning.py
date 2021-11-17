import numpy as np
import torch
from utils import *


def choose_criterion(criterion):
    if criterion == "random":
        return random_sampling
    elif criterion == "BALD":
        return bald
    elif criterion == "uncertainty_nn":
        return uncertainty_nn
    elif criterion == "uncertainty_gp":
        return uncertainty_gp


def bald():
    return 1


def random_sampling(model, train, query, test):
    n = len(query['pref'])
    return np.random.randint(0, n)


def uncertainty_nn(model, train, query, test):
    model.eval()
    x_query = query['x_duels']
    logistic_value = torch.zeros(len(query['pref']))
    for i in range(len(x_query)):
        x1 = torch.tensor(x_query[i][0])
        x2 = torch.tensor(x_query[i][1])

        out1, out2 = model(x1, x2)
        diff = torch.abs(out1 - out2)
        v = logistic_function(diff)
        logistic_value[i] = v
        # print(v)

    return torch.argmin(logistic_value)


def uncertainty_gp(model, train, query, test):
    x_query = query['x_duels']
    logistic_value = np.zeros(len(query['pref']))
    for i in range(len(x_query)):
        x1 = x_query[i][0]
        x2 = x_query[i][1]
        out1 = model.predict(x1.reshape(1, -1))
        out2 = model.predict(x2.reshape(1, -1))
        diff = np.abs(out1 - out2)
        v = logistic_function(diff)
        logistic_value[i] = v
    return np.argmin(logistic_value)
