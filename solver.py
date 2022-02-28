from model import PrefNet
from dataset import pref_dataset
from utils import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from GPro.preference import ProbitPreferenceGP
import copy
import active_learning
import torchbnn as bnn
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_nn(x_duels, pref, model=None):
    """
    trainer for preferential neural network
    :param x_duels: all the input pairs
    :param pref: ground truth labels provided by expert
    :param model: preferential neural network model
    :return: updated model after training
    """
    pref_set = pref_dataset(x_duels, pref)
    pref_train_loader = DataLoader(pref_set, batch_size=2, shuffle=True, drop_last=False)
    pref_net = PrefNet(x_duels[0][0].size).to(device) if model is None else model
    pref_net.double()

    criterion = torch.nn.NLLLoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

    optimizer = torch.optim.Adam(pref_net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.0001, T_max=20)

    # train with preference pairs
    for epoch in range(20):
        pref_net.train()
        train_loss = 0

        for idx, data in enumerate(pref_train_loader):
            x1 = data['x1']
            x2 = data['x2']

            pref = data['pref'].long()
            x1, x2, pref = x1.to(device), x2.to(device), pref.to(device)
            optimizer.zero_grad()

            output1, output2 = pref_net(x1, x2)

            # here for binary classification, log_softmax on the two outputs equal to our paper's connection function
            output = F.log_softmax(torch.hstack((output1, output2)), dim=1)

            loss = criterion(output, pref) + 0.1 * kl_loss(pref_net)
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        # print('[Epoch : %d] loss: %.3f' % (epoch + 1, train_loss / len(pref_train_loader)))
    return pref_net


def compute_nn_acc(model, test):
    """
    compute the preference accuracy for neural network
    :param model: current model
    :param test: test set
    :return: prediction accuracy
    """
    model.eval()
    x_test = test['x_duels']
    pref_test = test['pref']
    acc = 0
    n_mc = 10
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

    return acc


def active_train_nn(model, train0, query0, test, n_acq, al_criterion):
    """
    active learning control function for nn
    :param model: current model
    :param train0: train set
    :param query0: query set
    :param test: test set
    :param n_acq: active learning acquisition time
    :param al_criterion: active learning criterion
    :return: accuracy of preference prediction
    """
    model = copy.deepcopy(model)
    train = train0.copy()
    query = query0.copy()

    acc = np.zeros(n_acq + 1, )
    acc[0] = compute_nn_acc(model, test)
    print(0, acc[0])
    al_function = active_learning.choose_criterion(al_criterion)
    t = np.zeros(n_acq, )
    for i in range(n_acq):
        t1 = time.time()
        query_index = al_function(model, train, query, test)
        pref_q = query['pref'][query_index]

        train['x_duels'] = np.vstack((train['x_duels'], query['x_duels'][[query_index], :]))
        train['pref'] = np.hstack((train['pref'], pref_q))

        query['x_duels'] = np.delete(query['x_duels'], query_index, axis=0)
        query['pref'] = np.delete(query['pref'], query_index)

        model = train_nn(train['x_duels'], train['pref'], model)
        t2 = time.time()
        t[i] = str(t2-t1)
        acc[i+1] = compute_nn_acc(model, test)
        print(i + 1, acc[i + 1])

    return acc, t


def train_gp(x_duels, pref, model=None):
    """
    gp-based solver
    :param x_duels: all the input pairs
    :param pref: ground truth labels provided by expert
    :param model: current gp model
    :return: updated gp model
    """
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
    """
    compute the preference accuracy for gp-based model
    :param model: current model
    :param test: test set
    :return: prediction accuracy
    """
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
    return acc


def active_train_gp(model, train0, query0, test, n_acq, al_criterion):
    """
    active learning control function for gp
    :param model: current model
    :param train0: train set
    :param query0: query set
    :param test: test set
    :param n_acq: active learning acquisition time
    :param al_criterion: active learning criterion
    :return: accuracy of preference prediction
    """
    train = train0.copy()
    query = query0.copy()
    model = copy.deepcopy(model)
    acc = np.zeros(n_acq + 1, )
    acc[0] = compute_gp_acc(model, test)

    al_function = active_learning.choose_criterion(al_criterion)
    t = np.zeros(n_acq, )
    for i in range(n_acq):
        t1 = time.time()
        query_index = al_function(model, train, query, test)
        pref_q = query['pref'][query_index]

        train['x_duels'] = np.vstack((train['x_duels'], query['x_duels'][[query_index], :]))
        train['pref'] = np.hstack((train['pref'], pref_q))

        query['x_duels'] = np.delete(query['x_duels'], query_index, axis=0)
        query['pref'] = np.delete(query['pref'], query_index)

        model = train_gp(train['x_duels'], train['pref'], model)
        t2 = time.time()
        t[i] = str(t2 - t1)
        acc[i+1] = compute_gp_acc(model, test)
        print(i + 1, acc[i + 1])

    return acc, t



