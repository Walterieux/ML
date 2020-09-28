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
    w_toplot=[]
    w_toplot.append(w)
    
    for n_iter in range(max_iters):

        # 1) on pourrait rajouter une autre condition ici assez simple qui regarde l'écart entre deux différents w et si jamais 
        # la norme du gradient est petite on pourrait quitter, je mets le code en commentaire mais ça me parait le plus logique 
        # 2) est ce qu'on adapterait notre pas en fonction de où on est ? Dans le cours il parle de peut être faire -gamma/n_iter

        current_gradient = compute_gradient(y, tx, w, n)
        if np.linalg.norm(current_gradient) <= 1e-6:
            break
        w = w - gamma/(n_iter+1) * compute_gradient(y, tx, w, n)
        w_toplot.append(w)
    toplot= np.zeros((max_iters,1))
    for i in range(max_iters):
        toplot[i]=compute_loss(y, tx,w_toplot[i] )
    print(toplot)
    plt.plot(np.linspace(1,max_iters,max_iters),toplot)
    plt.scatter(np.linspace(1,max_iters,max_iters),toplot)
    plt.title("Error in term of number of iterations")
    plt.xlabel("number of iterations")
    plt.ylabel("Error")
    plt.yscale("log")
    return w, compute_loss(y, tx, w)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    n = len(y)

    rnd_sample = np.random.random_integers(0, n, max_iters)

    w = initial_w
    for n_iter in range(max_iters):
        subgradient = compute_gradient(y[rnd_sample[n_iter]], tx[rnd_sample[n_iter], :], w, n)
        if np.linalg.norm(subgradient) <= 1e-6:
            break
        w = w - gamma * subgradient
    return w, compute_loss(y, tx, w)
