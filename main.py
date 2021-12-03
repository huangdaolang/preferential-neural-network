import config
from utils import *
import solver
import numpy as np


def main(train_pair, query_pair, test_pair, n_acq):
    # get data
    train, query, test = get_boston_data(train_pair, query_pair, test_pair)

    acc_nn = np.zeros((10, 3, n_acq + 1))
    for i in range(10):
        model = solver.train_nn(train['x_duels'], train['pref'], model=None)
        acc_nn[i, 0, :] = solver.active_train_nn(model, train, query, test, n_acq, "random")
        acc_nn[i, 1, :] = solver.active_train_nn(model, train, query, test, n_acq, "uncertainty_nn")
        acc_nn[i, 2, :] = solver.active_train_nn(model, train, query, test, n_acq, "BALD_nn")
    acc_nn_std = np.std(acc_nn, axis=0)
    acc_nn_avg = np.mean(acc_nn, axis=0)
    print(acc_nn_avg)

    acc_gp = np.zeros((10, 3, n_acq + 1))
    for i in range(10):
        gp_model = solver.train_gp(train['x_duels'], train['pref'], model=None)
        acc_gp[i, 0, :] = solver.active_train_gp(gp_model, train, query, test, n_acq, "random")
        acc_gp[i, 1, :] = solver.active_train_gp(gp_model, train, query, test, n_acq, "uncertainty_gp")
        acc_gp[i, 2, :] = solver.active_train_gp(gp_model, train, query, test, n_acq, "BALD_gp")
    acc_gp_std = np.std(acc_gp, axis=0)
    acc_gp_avg = np.mean(acc_gp, axis=0)
    print(acc_gp_avg)

    plot_acc_trend(acc_nn_avg, acc_nn_std, acc_gp_avg, acc_gp_std, "final accuracy comparison.png")


if __name__ == "__main__":
    n_train_pairs = config.N_train_pair
    n_query_pairs = config.N_query_pair
    n_test_pairs = config.N_test_pair
    n_acquire = config.N_acquire
    main(n_train_pairs, n_query_pairs, n_test_pairs, n_acquire)

