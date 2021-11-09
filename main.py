import config
from utils import forrester_function, logistic_function, PrefLoss_Forrester
from model import PrefNet_Forrester
from dataset import pref_dataset
import numpy as np
import pandas as pd
import torch
import itertools
import random

from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
from GPro.preference import ProbitPreferenceGP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        train_index = random.sample(range(0, len(pairs)), nb)
        x_duels = np.array([[x[pairs[index][0]], x[pairs[index][1]]] for index in train_index]).reshape(-1, 2)
        pref = []
        for index in train_index:
            pref.append(1) if y[pairs[index][0]] < y[pairs[index][1]] else pref.append(0)

        pref_set = pref_dataset(x_duels, pref)
        pref_train_loader = DataLoader(pref_set, batch_size=50, shuffle=True, drop_last=False)

        pref_net = PrefNet_Forrester().to(device)
        pref_net.double()
        criterion = PrefLoss_Forrester()
        optimizer = torch.optim.Adam(pref_net.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.0001, T_max=20)
        for epoch in range(300):
            pref_net.train()
            train_loss = 0
            for idx, data in enumerate(pref_train_loader):
                x1 = data['x1']
                x2 = data['x2']

                # pref = (torch.rand(size=[len(data['pref'])]) < data['pref']) * 1  # random process
                pref = data['pref']
                x1, x2, pref = x1.to(device), x2.to(device), pref.to(device)
                optimizer.zero_grad()
                output1, output2 = pref_net(x1, x2)
                loss = criterion(output1, output2, pref)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            # print('[Epoch : %d] loss: %.3f' % (epoch + 1, train_loss / len(pref_train_loader)))

        out, _ = pref_net(nn_x_test, nn_x_test)
        out = out.detach().numpy()

        acc = 0
        for pair in pairs:
            if y[pair[0]] > y[pair[1]] and out[pair[0]] > out[pair[1]]:
                acc += 1
            if y[pair[0]] < y[pair[1]] and out[pair[0]] < out[pair[1]]:
                acc += 1
        acc = acc / len(pairs)

        if acc > 0.75:
            nn_acc_repeat.append(acc)
    acc_avg = np.mean(nn_acc_repeat)
    print("nn: ", acc_avg)
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
    print("gp: ", gp_acc_tmp)
    gp_acc.append(gp_acc_tmp)
    plt.plot(nb_list, gp_acc, c="red")
    plt.plot(nb_list, nn_acc, c="green")
    plt.show()
    # plt.plot(x, out/2)
    # plt.plot(x, y, c="red", label="True")
    # plt.scatter(x[np.argmin(out)], np.min(out/2), marker="*", c="black")
    # plt.scatter(x[np.argmin(y)], np.min(y), marker="^", c="blue")
    # plt.show()
