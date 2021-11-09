from model import PrefNet_Forrester
from dataset import pref_dataset
import random
import numpy as np
from utils import PrefLoss_Forrester
import torch
from torch.utils.data import DataLoader
from GPro.preference import ProbitPreferenceGP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_nn(x, y, x_test, pairs, nb):
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

    out, _ = pref_net(x_test, x_test)
    out = out.detach().numpy()

    acc = 0
    for pair in pairs:
        if y[pair[0]] > y[pair[1]] and out[pair[0]] > out[pair[1]]:
            acc += 1
        if y[pair[0]] < y[pair[1]] and out[pair[0]] < out[pair[1]]:
            acc += 1
    acc = acc / len(pairs)

    return acc


def train_gp(x, y, m, x_test, pairs, nb):
    random.shuffle(m)
    M = m[:nb]
    gpr = ProbitPreferenceGP()
    gpr.fit(x, M, f_prior=None)
    gp_pred = gpr.predict(x_test)
    acc = 0
    for pair in pairs:
        if y[pair[0]] > y[pair[1]] and gp_pred[pair[0]] < gp_pred[pair[1]]:
            acc += 1
        if y[pair[0]] < y[pair[1]] and gp_pred[pair[0]] > gp_pred[pair[1]]:
            acc += 1
    acc = acc / len(pairs)
    return acc
