from model import PrefNet
from dataset import pref_dataset, inducing_dataset
import random
import numpy as np
from utils import PrefLoss_Forrester, forrester_function, plot_function_shape, logistic_function
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from GPro.preference import ProbitPreferenceGP
import copy
import active_learning
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torchbnn as bnn
# from torchhk import transform_model


def train_nn(x_duels, pref, model=None):
    pref_set = pref_dataset(x_duels, pref)
    pref_train_loader = DataLoader(pref_set, batch_size=10, shuffle=True, drop_last=False)
    pref_net = PrefNet(x_duels[0][0].size).to(device) if model is None else model
    pref_net.double()

    # criterion = PrefLoss_Forrester()
    criterion = torch.nn.NLLLoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    optimizer = torch.optim.Adam(pref_net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.0001, T_max=20)

    for epoch in range(100):
        pref_net.train()
        train_loss = 0
        # train with preference pairs
        for idx, data in enumerate(pref_train_loader):
            x1 = data['x1']
            x2 = data['x2']
            # pref = (torch.rand(size=[len(data['pref'])]) < data['pref']) * 1  # random process
            # pref = data['pref'].double()
            pref = data['pref'].long()
            x1, x2, pref = x1.to(device), x2.to(device), pref.to(device)
            optimizer.zero_grad()
            output1, output2 = pref_net(x1, x2)

            output = F.log_softmax(torch.hstack((output1, output2)), dim=1)

            # output = logistic_function(output1-output2).flatten()
            loss = criterion(output, pref) + 0.1 * kl_loss(pref_net)

            # loss = criterion(output1, output2, pref)
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        # print('[Epoch : %d] loss: %.3f' % (epoch + 1, train_loss / len(pref_train_loader)))
    return pref_net


def compute_nn_acc(model, test):
    model.eval()
    x_test = test['x_duels']
    pref_test = test['pref']
    acc = 0
    n_mc = 3
    for i in range(len(x_test)):
        x1 = torch.tensor(x_test[i][0])
        x2 = torch.tensor(x_test[i][1])
        pref = pref_test[i]
        out = torch.zeros((n_mc, 2))
        for n in range(n_mc):
            out[n, 0], out[n, 1] = model(x1, x2)
        pred = torch.mean(out, dim=0)
        out1 = pred[0]
        out2 = pred[1]
        if pref == 1 and out1 < out2:
            acc += 1
        if pref == 0 and out1 > out2:
            acc += 1
    acc = acc / len(x_test)
    print("nn", acc)
    return acc


def active_train_nn(model, train0, query0, test, n_acq, al_criterion):
    model = copy.deepcopy(model)
    train = train0.copy()
    query = query0.copy()

    acc = np.zeros(n_acq + 1, )
    acc[0] = compute_nn_acc(model, test)

    al_function = active_learning.choose_criterion(al_criterion)

    for i in range(n_acq):
        query_index = al_function(model, train, query, test)
        pref_q = query['pref'][query_index]

        train['x_duels'] = np.vstack((train['x_duels'], query['x_duels'][[query_index], :]))
        train['pref'] = np.hstack((train['pref'], pref_q))

        query['x_duels'] = np.delete(query['x_duels'], query_index, axis=0)
        query['pref'] = np.delete(query['pref'], query_index)

        model = train_nn(train['x_duels'], train['pref'], model)

        acc[i+1] = compute_nn_acc(model, test)

    return acc


def train_gp(x_duels, pref, model=None):
    x_train = []
    M_train = []
    for i in range(len(x_duels)):
        x_train.append(x_duels[i][0])
        x_train.append(x_duels[i][1])
        M_train.append([2 * i, 2 * i + 1]) if pref[i] == 1 else M_train.append([2 * i + 1, 2 * i])

    gpr = ProbitPreferenceGP()
    gpr.fit(x_train, M_train, f_prior=None)
    return gpr


def compute_gp_acc(model, test):
    x_test = test['x_duels']
    pref_test = test['pref']
    acc = 0
    for i in range(len(x_test)):
        x1 = x_test[i][0]
        x2 = x_test[i][1]
        pref = pref_test[i]
        out1 = model.predict(x1.reshape(1, -1))
        out2 = model.predict(x2.reshape(1, -1))
        if pref == 1 and out1 > out2:
            acc += 1
        if pref == 0 and out1 < out2:
            acc += 1
    acc = acc / len(x_test)
    print("gp", acc)
    return acc


def active_train_gp(model, train0, query0, test, n_acq, al_criterion):
    train = train0.copy()
    query = query0.copy()
    model = copy.deepcopy(model)
    acc = np.zeros(n_acq + 1, )
    acc[0] = compute_gp_acc(model, test)

    al_function = active_learning.choose_criterion(al_criterion)

    for i in range(n_acq):
        query_index = al_function(model, train, query, test)
        pref_q = query['pref'][query_index]

        train['x_duels'] = np.vstack((train['x_duels'], query['x_duels'][[query_index], :]))
        train['pref'] = np.hstack((train['pref'], pref_q))

        query['x_duels'] = np.delete(query['x_duels'], query_index, axis=0)
        query['pref'] = np.delete(query['pref'], query_index)

        model = train_gp(train['x_duels'], train['pref'], model)

        acc[i+1] = compute_gp_acc(model, test)

    return acc



