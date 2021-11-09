import config
from utils import forrester_function
import solver
import numpy as np
import torch
import itertools
import random
import matplotlib.pyplot as plt
from GPro.preference import ProbitPreferenceGP
x = np.linspace(0, 1, config.N_points)
nn_x_test = torch.from_numpy(x).reshape(-1, 1)
gp_x_test = x.reshape(-1, 1)
y = forrester_function(x)
pairs = list(itertools.permutations(range(len(x)), 2))
random.shuffle(pairs)

gp_x = x.reshape(-1, 1)
gp_y = []
for a in range(len(x)):
    for b in range(len(x)):
        if y[a] < y[b]:
            gp_y.append([a, b])
        elif y[a] > y[b]:
            gp_y.append([b, a])

nn_acc = []
gp_acc = []
nb_list = []
for nb in range(config.N_train_pair[0], config.N_train_pair[1], 20):
    nb_list.append(nb)
    nn_acc_repeat = []
    # repeat 10 times to get stable results
    for i in range(10):
        acc = solver.train_nn(x, y, nn_x_test, pairs, nb)
        if acc > 0.75:
            nn_acc_repeat.append(acc)
    acc_avg = np.mean(nn_acc_repeat)
    print("nn accuracy under {} points: {}".format(nb, acc_avg))
    nn_acc.append(acc_avg)

    random.shuffle(gp_y)
    M = gp_y[:nb]
    gpr = ProbitPreferenceGP()
    gpr.fit(gp_x, M, f_prior=None)
    gp_pred = gpr.predict(gp_x_test)
    gp_acc_tmp = 0
    for pair in pairs:
        if y[pair[0]] > y[pair[1]] and gp_pred[pair[0]] < gp_pred[pair[1]]:
            gp_acc_tmp += 1
        if y[pair[0]] < y[pair[1]] and gp_pred[pair[0]] > gp_pred[pair[1]]:
            gp_acc_tmp += 1
    gp_acc_tmp = gp_acc_tmp / len(pairs)
    print("gp accuracy under {} points: {}".format(nb, gp_acc_tmp))
    gp_acc.append(gp_acc_tmp)
    plt.plot(nb_list, gp_acc, c="red")
    plt.plot(nb_list, nn_acc, c="green")
    plt.show()
    # plt.plot(x, out/2)
    # plt.plot(x, y, c="red", label="True")
    # plt.scatter(x[np.argmin(out)], np.min(out/2), marker="*", c="black")
    # plt.scatter(x[np.argmin(y)], np.min(y), marker="^", c="blue")
    # plt.show()
