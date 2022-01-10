import numpy as np
import torch
import scipy as sp
from utils import *


def choose_criterion(criterion):
    if criterion == "random":
        return random_sampling
    elif criterion == "nn_bald":
        return nn_bald
    elif criterion == "gp_bald":
        return gp_bald
    elif criterion == "nn_lc":
        return nn_least_confident
    elif criterion == "gp_lc":
        return gp_least_confident
    elif criterion == "nn_me":
        return nn_maximize_entropy
    elif criterion == "gp_me":
        return gp_maximize_entropy


def random_sampling(model, train, query, test):
    n = len(query['pref'])
    return np.random.randint(0, n)


def nn_least_confident(model, train, query, test):
    model.eval()
    x_query = query['x_duels']
    confidence = torch.zeros(len(query['pref']))
    n_mc = 5
    for i in range(len(x_query)):
        x1 = torch.tensor(x_query[i][0])
        x2 = torch.tensor(x_query[i][1])
        out = torch.zeros((n_mc, 2))
        for n in range(n_mc):
            out[n, 0], out[n, 1] = model(x1, x2)
        pred = torch.mean(out, dim=0)
        out1 = pred[0]
        out2 = pred[1]
        # out1, out2 = model(x1, x2)
        diff = torch.abs(out1 - out2)
        v = logistic_function(diff)
        confidence[i] = v

    return torch.argmin(confidence)


def nn_maximize_entropy(model, train, query, test):
    x_query = query['x_duels']
    entropy = torch.zeros(len(query['pref']))
    for i in range(len(x_query)):
        x1 = torch.tensor(x_query[i][0])
        x2 = torch.tensor(x_query[i][1])

        out1, out2 = model(x1, x2)

        diff_0 = logistic_function(out1 - out2)
        diff_1 = logistic_function(out2 - out1)
        entropy[i] = -(diff_0 * torch.log2(diff_0) + diff_1 * torch.log2(diff_1))

    return torch.argmax(entropy)


def nn_bald(model, train, query, test):
    x_query = query['x_duels']
    dropout_iterations = 20  # forward time, 100 in Gal's paper
    x1 = torch.tensor(x_query[:, 0, :])
    x2 = torch.tensor(x_query[:, 1, :])

    score_all = np.zeros(shape=(x1.shape[0], 2))
    all_entropy_dropout = np.zeros(shape=x1.shape[0])
    for t in range(dropout_iterations):
        out1, out2 = model(x1, x2)
        diff = out2 - out1
        prob_1 = logistic_function(diff)
        prob_0 = 1 - prob_1
        score = torch.cat((prob_0, prob_1), 1)
        score = score.detach().numpy()
        # print(score)
        score_all += score
        score_log = np.log2(score)

        entropy_compute = - np.multiply(score, score_log)
        entropy_per_dropout = np.sum(entropy_compute, axis=1)

        all_entropy_dropout += entropy_per_dropout
    score_all += np.finfo(float).eps
    Avg_Pi = np.divide(score_all, dropout_iterations)
    Log_Avg_Pi = np.log2(Avg_Pi)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

    G_X = Entropy_Average_Pi

    F_X = np.divide(all_entropy_dropout, dropout_iterations)

    U_X = (G_X + F_X).flatten()

    return np.nanargmax(U_X)


def gp_least_confident(model, train, query, test):
    x_query = query['x_duels']
    confidence = np.zeros(len(query['pref']))
    for i in range(len(x_query)):
        x1 = x_query[i][0]
        x2 = x_query[i][1]
        out1 = model.predict(x1.reshape(1, -1))
        out2 = model.predict(x2.reshape(1, -1))
        diff = np.abs(out1 - out2)
        v = logistic_function(diff)
        confidence[i] = v
    return np.argmin(confidence)


def gp_maximize_entropy(model, train, query, test):
    x_query = query['x_duels']
    entropy = np.zeros(len(query['pref']))
    for i in range(len(x_query)):
        x1 = x_query[i][0]
        x2 = x_query[i][1]
        out1 = model.predict(x1.reshape(1, -1))
        out2 = model.predict(x2.reshape(1, -1))

        diff_0 = logistic_function(out1 - out2)
        diff_1 = logistic_function(out2 - out1)
        entropy[i] = -(diff_0 * np.log2(diff_0) + diff_1 * np.log2(diff_1))
    return np.argmax(entropy)


def gp_bald(model, train, query, test):
    def h(p):
        return -p*np.log2(p) - (1-p)*np.log2(1-p)
    C = np.sqrt(np.pi * np.log(2)/2)
    x_query = query['x_duels']
    duel_bald_score = np.zeros(len(query['pref']))
    for i in range(len(x_query)):
        x1 = x_query[i][0]
        x2 = x_query[i][1]
        m1, s1 = model.predict(x1.reshape(1, -1), return_y_std=True)
        m2, s2 = model.predict(x2.reshape(1, -1), return_y_std=True)
        x1_score = h(sp.special.ndtr(m1/np.sqrt(s1**2+1))) - C*np.exp(-m1**2/(2*(s1**2+C**2)))/np.sqrt(s1**2+C**2)
        x2_score = h(sp.special.ndtr(m2 / np.sqrt(s2 ** 2 + 1))) - C * np.exp(
            -m2 ** 2 / (2 * (s2 ** 2 + C ** 2))) / np.sqrt(s2 ** 2 + C ** 2)

        duel_bald_score[i] = x1_score + x2_score
    return np.nanargmax(duel_bald_score)