import config
from utils import *
import solver
import numpy as np
import sys
import os


def main(train_pair, query_pair, test_pair, n_acq, index):
    """
    simulation start from here
    :param train_pair: number of initial train pair
    :param query_pair: D_pool in paper
    :param test_pair: test set for evaluation performance
    :param n_acq: active learning acquisition time
    :param index: record simulation index
    """
    root_name = 'Sim/' + config.dataset
    if not os.path.exists(root_name):
        os.mkdir(root_name)

    # retrieve data
    train, query, test = get_data(config.dataset, train_pair, query_pair, test_pair, index)

    # neural network part for three different active learning strategies
    acc_nn = np.zeros((3, n_acq + 1))
    model = solver.train_nn(train['x_duels'], train['pref'], model=None)
    acc_nn[0, :] = solver.active_train_nn(model, train, query, test, n_acq, "random")
    acc_nn[1, :] = solver.active_train_nn(model, train, query, test, n_acq, "nn_lc")
    acc_nn[2, :] = solver.active_train_nn(model, train, query, test, n_acq, "nn_bald")

    print('Saving results for nn...')
    np.save(root_name + '/nn_' + str(index) + '.npy', acc_nn)
    print(acc_nn)

    # gp part for three different active learning strategies
    acc_gp = np.zeros((3, n_acq + 1))
    gp_model = solver.train_gp(train['x_duels'], train['pref'], model=None)
    acc_gp[0, :] = solver.active_train_gp(gp_model, train, query, test, n_acq, "random")
    acc_gp[1, :] = solver.active_train_gp(gp_model, train, query, test, n_acq, "gp_lc")
    acc_gp[2, :] = solver.active_train_gp(gp_model, train, query, test, n_acq, "gp_bald")

    print('Saving results for gp...')
    np.save(root_name + '/gp_' + str(index) + '.npy', acc_gp)
    print(acc_gp)


if __name__ == "__main__":
    n_train_pairs = config.N_train_pair
    n_query_pairs = config.N_query_pair
    n_test_pairs = config.N_test_pair
    n_acquire = config.N_acquire
    sim = int(sys.argv[1])
    main(n_train_pairs, n_query_pairs, n_test_pairs, n_acquire, sim)
