import config
from utils import forrester_function, plot_acc_trend
import solver
import numpy as np
import torch
import itertools
import random
import time

# generate data for nn
x = np.linspace(0, 1, config.N_points)
nn_x_test = torch.from_numpy(x).reshape(-1, 1)
y = forrester_function(x)
pairs = list(itertools.permutations(range(len(x)), 2))
random.shuffle(pairs)

# generate data for gp
gp_x = x.reshape(-1, 1)
gp_x_test = x.reshape(-1, 1)
gp_m = []
for a in range(len(x)):
    for b in range(len(x)):
        if y[a] < y[b]:
            gp_m.append([a, b])
        elif y[a] > y[b]:
            gp_m.append([b, a])


nn_acc_list = []
gp_acc_list = []
nb_list = []
for nb in range(config.N_train_pair[0], config.N_train_pair[1], 20):
    nb_list.append(nb)

    # train nn
    nn_acc_repeat = []
    time1 = time.time()
    for _ in range(10):
        nn_acc = solver.train_nn(x, y, nn_x_test, pairs, nb)
        if nn_acc > 0.75:
            nn_acc_repeat.append(nn_acc)
    time2 = time.time()
    nn_time = (time2 - time1) / 10
    nn_acc_avg = np.mean(nn_acc_repeat)
    nn_acc_list.append(nn_acc_avg)
    print("nn accuracy under {} points: {}, time takes {}".format(nb, nn_acc_avg, nn_time))

    # train gp
    gp_acc_repeat = []
    time1 = time.time()
    for _ in range(10):
        gp_acc = solver.train_gp(gp_x, y, gp_m, gp_x_test, pairs, nb)
        gp_acc_repeat.append(gp_acc)
    time2 = time.time()
    gp_time = (time2 - time1) / 10
    gp_acc_avg = np.mean(gp_acc_repeat)
    gp_acc_list.append(gp_acc_avg)
    print("gp accuracy under {} points: {}, time takes {}".format(nb, gp_acc_avg, gp_time))

plot_acc_trend(nb_list, gp_acc_list, nn_acc_list)



