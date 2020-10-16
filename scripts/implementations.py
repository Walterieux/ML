# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:47:51 2020

@author: jeang
"""
import numpy as np
from matplotlib import pyplot as plt
from proj1_helpers import *
import seaborn as sns


def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    For this we will calculate MSE
    """

    total = 0.5 * np.mean(np.square(y - np.dot(tx, w)))
    return total


def compute_loss_MAE(y, tx, w):
    total = np.mean(np.abs(y - np.dot(tx, w)))
    return total

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    # ***************************************************


def compute_gradient(y, tx, w, N):
    # basically we have gradient L(w) = -1/N X^Te where e=(y-Xw)
    gradient = -1 / N * np.dot(np.transpose(tx), y - np.dot(tx, w))
    return gradient


def compute_gradient_MAE(y, tx, w, N):
    # on a 1/N sum_1^N |y-x^_ntw|
    # donc le gradient ça devrait être -1/N * X^T sign(y-x^t_nw)
    gradient = -1 / N * np.dot(np.transpose(tx), np.sign(y - np.dot(tx, w)))
    return gradient


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    n = len(y)
    print("max_iters = " + repr(max_iters))
    w = initial_w
    w_toplot = []
    w_toplot.append(w)

    for n_iter in range(max_iters):
        current_gradient = compute_gradient(y, tx, w, n)
        if np.linalg.norm(current_gradient) <= 1e-6:
            break
        w = w - gamma / (n_iter + 1) * compute_gradient(y, tx, w, n)
        w_toplot.append(w)

    toplot = np.zeros((max_iters, 1))
    for i in range(max_iters):
        toplot[i] = compute_loss(y, tx, w_toplot[i])
    print(toplot)
    plt.plot(np.linspace(1, max_iters, max_iters), toplot)
    plt.scatter(np.linspace(1, max_iters, max_iters), toplot)
    plt.title("Error in term of number of iterations")
    plt.xlabel("number of iterations")
    plt.ylabel("Error")
    plt.yscale("log")

    return w, compute_loss(y, tx, w)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    n = len(y)

    rnd_sample = np.random.random_integers(0, n - 1, max_iters)

    w = initial_w
    for n_iter in range(max_iters):
        subgradient = compute_gradient(y[rnd_sample[n_iter]], tx[rnd_sample[n_iter], :], w, n)
        if np.linalg.norm(subgradient) <= 1e-6:
            break
        w = w - gamma * subgradient

    return w, compute_loss(y, tx, w)


def least_squares(y, tx):
    x_transp = np.transpose(tx)
    w = np.linalg.solve(np.dot(x_transp, tx), np.dot(x_transp, y))
    return w, compute_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    x_transp = np.transpose(tx)

    A = np.dot(x_transp, tx)
    B = np.dot(lambda_ * 2 * len(y), np.identity(x_transp.shape[0]))

    y1 = np.dot(x_transp, y)
    w = np.linalg.solve(A + B, y1)

    return w, compute_loss(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    n = len(y)
    w = initial_w

    for n_iter in range(max_iters):
        gradient = np.dot(np.transpose(tx), sigma(np.dot(tx, w) - y))
        if np.linalg.norm(gradient) <= 1e-6:
            break
        w = w - gamma / (n_iter + 1) * gradient

    return w, compute_loss(y, tx, w)


def sigma(x):
    return np.exp(x) / (1 + np.exp(x))


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # TODO
    w = 0

    return w, compute_loss(y, tx, w)


def sorted_by_variance(tx):
    return np.argsort(np.std(tx, axis=0))


def test_find_degree(y_train, y_test, tx_train, tx_test):
    degree1 = np.linspace(7, 9, 3, dtype=int)
    degree2 = np.linspace(7, 9, 3, dtype=int)
    degree3 = np.linspace(7, 9, 3, dtype=int)
    degree4 = np.linspace(7, 9, 3, dtype=int)
    degree5 = np.linspace(7, 9, 3, dtype=int)
    best_array = np.zeros(5, dtype=int)
    min_value = 10000
    # len_degree2=len(degree2)
    # len_degree3=len()
    for i in degree1:
        for j in degree2:
            for k in degree3:
                for l in degree4:
                    for m in degree5:
                        # print("i : " , i , "j : ", k , "k :" ,k, "l : ", l , "m : ", m )
                        current_tx = build_poly_variance(tx_train, 0, i, j, k, l, m)
                        weights, loss = ridge_regression(y_train, current_tx, 0.0)
                        tx_test_poly = build_poly_variance(tx_test, 0, i, j, k, l, m)
                        # y_pred = predict_labels(weights, tx_test_poly)
                        current_val = compute_loss(y_test, tx_test_poly, weights)
                        if current_val <= min_value:
                            min_value = current_val
                            print("i : ", i, "j : ", k, "k :", k, "l : ", l, "m : ", m)

                            print("current_val : ", current_val)
                            best_array = np.array([i, j, k, l, m])
    return min_value, best_array


def build_poly_variance(tx, allparam, degree1=0, degree2=3, degree3=5, degree4=4, degree5=5):
    # classification en fonction de la variance de chaque paramètre:
    nrows, ncols = np.shape(tx)

    if allparam == 0:
        params = np.array([degree1, degree2, degree3, degree4, degree5])
        nb_gr_vr = ncols // 5

    # calcul du nombre total de colonnes, on ajoute 1 étant donné qu'on impose un terme indépendant.
    ncol_tot = np.sum(params * nb_gr_vr) + 1 + params[-1] * (ncols % 5)

    newx = np.zeros((nrows, ncol_tot))
    current = 0
    # list_nb_gr_vr=np.full((len(params),nb_gr_vr,dtype=int)
    # list_nb_gr_vr[-1]+=nb_gr_vr % 5
    for counter, param in enumerate(params[:-1]):
        # print("param :" , param)
        # print("current : ", current)
        # print("current + nb_gr_vr * param",current + nb_gr_vr * param)
        newx[:, current:current + nb_gr_vr * param] = build_poly(tx[:, (counter) * nb_gr_vr:(counter + 1) * nb_gr_vr],
                                                                 param)
        current = nb_gr_vr * param + current
    newx[:, current:-1] = build_poly(tx[:, (counter + 1) * nb_gr_vr:], params[-1])
    newx[:, -1] = 1
    # print(newx)
    return newx


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=1 up to j=degree."""
    x_flat = np.ndarray.flatten(x)  # ça va juste reshape une array en 1d array.

    rows, cols = np.indices((x_flat.shape[0], degree))  #
    res = x_flat[rows]  # ça ça va juste augmenter le nombre de x_flat ça va le duppliquer degree fois.

    res[rows, cols] = res[rows, cols] ** (cols + 1)  # mise à l'exposant.
    res = np.reshape(res, (x.shape[0], -1), 'A')
    return res


def variance_half_max_index(tx):
    sorted_variance = np.argsort(np.std(tx, axis=0))
    return sorted_variance[len(sorted_variance) // 5:]


# split data in percentage to train and 1-percentage to test
def data_train_test(y, tx, percentage):
    len_y = len(y)
    len_test = int(percentage * len_y)
    print(len_test)
    y_train = y[:len_test]
    y_test = y[len_test:]
    tx_train = tx[:len_test, ]
    tx_test = tx[len_test:, ]
    return y_train, y_test, tx_train, tx_test


def clean_data(tx):
    std_data = np.std(tx, axis=0)
    mean_data = np.mean(tx, axis=0)

    clustered_data = np.concatenate((mean_data - std_data < tx, tx < mean_data + std_data), axis=1)

    return tx[np.all(clustered_data, axis=1), :]


"""def standardize(x):
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data"""


# change value which are equal to -999 to 0 TODO (maybe the mean could be an idea too.. certanly a better idea)
# !!!!! Try it !!
def replace_999_data_elem(tx):
    return np.where(tx == -999, 0, tx)


""" retour=np.concatenate((tx,tx_test),axis=0)
    print(np.shape(retour))
    retour1=np.copy(retour)
    retour1=standardize(retour1)
    retour1[retour==-999]=0.0
    retour1=standardize(retour1)
    retour1[retour==-999]=0.0
    
    return retour[:n_rows,],retour[n_rows:,]"""


# Input: tX (data of features, isBinary(boolean) which indicates if we want values or only boolean higher than 0.9 or not
# Output: correlation matrix, (to know: correlation matrix indicates if there is a linear link btw two features, how n
#         near from, how higher the correlation)
def calculateCovariance(tX, isBinary):
    covariance = np.cov(tX, rowvar=False, bias=True)
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    if isBinary:
        return np.where(correlation >= 0.9, 1, 0)
    correlation[covariance == 0] = 0
    return correlation


# calculation unbiased covariance matrix with y, it does not have a lot of impact it would be too easy if there was
# linear link btw y and one  of the features
def calculateCovariance_y_tX(tX, y):
    y_tX = np.concatenate((np.transpose([y]), tX), axis=1)
    cov_y_tX = calculateCovariance(y_tX, 1)
    array_of_corr = np.where(cov_y_tX[:, 0] == 1)
    return array_of_corr


# get the features that are uncorrelated, it means it deletes the features that are too much correlated with other
# and discard them
def get_uncorellated_features(tX):
    binary_covariance = calculateCovariance(tX, 1)
    n_rows, n_cols = np.shape(binary_covariance)
    columns = np.full((n_rows,), True, dtype=bool)

    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            if binary_covariance[i, j] == 1:
                if columns[j]:
                    columns[j] = False
    return np.where(columns is True)[0]
