# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:47:51 2020

@author: jeang
"""
import numpy as np
from matplotlib import pyplot as plt


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

        # 1) on pourrait rajouter une autre condition ici assez simple qui regarde l'écart entre deux différents w et si jamais 
        # la norme du gradient est petite on pourrait quitter, je mets le code en commentaire mais ça me parait le plus logique 
        # 2) est ce qu'on adapterait notre pas en fonction de où on est ? Dans le cours il parle de peut être faire -gamma/n_iter

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


def build_poly_variance(tx, allparam, param1=0, param2=3, param3=5):
    # classification en fonction de la variance de chaque paramètre:
    sorted_variance = np.argsort(np.std(tx, axis=0))
    x = tx[:, sorted_variance]
    nrows, ncols = np.shape(x)

    if allparam == 0:
        nb_gr_vr = ncols // 3
        params = np.array([0, 3, 3])
    else:
        # nb_gr_vr=ncols
        # params=np.array([0,0,0,0,2,2,3,3,3,3,4,5,5,5,5])
        # print(len(params))
        nb_gr_vr = ncols // 10
        params = np.array([0, 0, 0, 2, 3, 3, 5, 6, 6, 6])

    # calcul du nombre total de colonnes, on ajoute 1 étant donné qu'on impose un terme indépendant.
    ncol_tot = np.sum(params * nb_gr_vr)

    newx = np.zeros((nrows, ncol_tot))

    current = 0

    for counter, param in enumerate(params):
        print(counter)
        newx[:, current:current + nb_gr_vr * param] = build_poly(x[:, (counter) * nb_gr_vr:(counter + 1) * nb_gr_vr],
                                                                 param)
        current = nb_gr_vr * param + current
    # newx[:,-1]=1

    return newx


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    x_flat = np.ndarray.flatten(x)  # ça va juste reshape une array en 1d array.

    rows, cols = np.indices((x_flat.shape[0], degree))  #
    res = x_flat[rows]  # ça ça va juste augmenter le nombre de x_flat ça va le duppliquer degree fois.

    res[rows, cols] = res[rows, cols] ** (cols + 1)  # mise à l'exposant.
    res = np.reshape(res, (x.shape[0], -1), 'A')
    return res


def ridge_regression(y, tx, lambda_):
    x_transp = np.transpose(tx)

    A = np.dot(x_transp, tx)

    B = np.dot(lambda_ * 2 * len(y), np.identity(x_transp.shape[0]))

    y1 = np.dot(x_transp, y)

    w = np.linalg.solve(A + B, y1)

    return w, compute_loss(y, tx, w)


def least_square(y, tx):
    X_transp = np.transpose(tx)

    w = np.linalg.solve(np.dot(X_transp, tx), np.dot(X_transp, y))
    return w, compute_loss(y, tx, w)


def variance_half_max_index(tx):
    sorted_variance = np.argsort(np.std(tx, axis=0))
    return sorted_variance[len(sorted_variance) // 5:]


def clean_data(tx):
    # standard deviation de chaque feature

    std_data = np.std(tx, axis=0)
    # moyenne de chaque feature

    mean_data = np.mean(tx, axis=0)

    # deux arrays qui contiennent mean-std_deviation (logiquement on devrait avoir 95% de nos données
    # là dedans mais là c'est en 1d du coup on en aura moins nécessairement..
    mean_below_std = mean_data - std_data
    mean_up_std = mean_data + std_data

    # deux matrices de la taille de tx qui contiennent des booléans voir si chaque élément est bien dans
    # le cluster. 'première au dessus, l'autre en dessous.

    uper_std = np.less(mean_below_std, tx)
    lower_std = np.greater(mean_up_std, tx)

    # matrice qui contient des boolean qui indique ici si c'est dans le cluster ou pas cette fois.
    clustered_data = (uper_std * lower_std)

    return tx[np.all(clustered_data, axis=1), :]
