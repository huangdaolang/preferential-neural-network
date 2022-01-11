import config
from utils import *
import solver
import numpy as np
import sys
import os


def main(train_pair, query_pair, test_pair, n_acq, seed):
    train, query, test = get_data(config.dataset, train_pair, query_pair, test_pair, seed)

    acc_nn = np.zeros((3, n_acq + 1))

    model = solver.train_nn(train['x_duels'], train['pref'], model=None)
    acc_nn[0, :] = solver.active_train_nn(model, train, query, test, n_acq, "random")
    acc_nn[1, :] = solver.active_train_nn(model, train, query, test, n_acq, "nn_lc")
    acc_nn[2, :] = solver.active_train_nn(model, train, query, test, n_acq, "nn_bald")

    print('Saving results for nn...')
    root_name = 'Sim/' + config.dataset
    if not os.path.exists(root_name):
        os.mkdir(root_name)
    np.save(root_name + '/nn_' + str(seed) + '.npy', acc_nn)
    print(acc_nn)

    acc_gp = np.zeros((3, n_acq + 1))
    gp_model = solver.train_gp(train['x_duels'], train['pref'], model=None)
    acc_gp[0, :] = solver.active_train_gp(gp_model, train, query, test, n_acq, "random")
    acc_gp[1, :] = solver.active_train_gp(gp_model, train, query, test, n_acq, "gp_lc")
    acc_gp[2, :] = solver.active_train_gp(gp_model, train, query, test, n_acq, "gp_bald")

    print('Saving results for gp...')
    root_name = 'Sim/' + config.dataset
    if not os.path.exists(root_name):
        os.mkdir(root_name)
    np.save(root_name + '/gp_' + str(seed) + '.npy', acc_gp)
    print(acc_gp)

    # fig_name = "al_acc_compare_" + config.dataset + ".png"
    # plot_acc_trend(acc_nn_avg, acc_nn_std, acc_gp_avg, acc_gp_std, fig_name)


if __name__ == "__main__":
    n_train_pairs = config.N_train_pair
    n_query_pairs = config.N_query_pair
    n_test_pairs = config.N_test_pair
    n_acquire = config.N_acquire
    sim = int(sys.argv[1])
    main(n_train_pairs, n_query_pairs, n_test_pairs, n_acquire, sim)
