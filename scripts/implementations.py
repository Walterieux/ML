# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:47:51 2020

@author: jeang
"""
import numpy as np
# TODO remove: only numpy is allowed
from matplotlib import pyplot as plt
import seaborn as sns

from proj1_helpers import *


#def loss_classification(y,tx,w):
    

def compute_loss_binary(y_test,tx_test,w_train): 
    y=np.where(y_test==-1,0,1)
    #y=y_test
    predicted_y=np.dot(tx_test,w_train)
    #predicted_y=np.where(predicted_y>0,1,-1)
    predicted_y=np.where(predicted_y>0.5,1,0)

    sum_y=predicted_y-y
    return (1-np.count_nonzero(sum_y)/len(sum_y))
    
def compute_loss(y, tx, w):
    """Calculate the loss using mean squared error (MSE)"""

    return 0.5 * np.mean(np.square(y - np.dot(tx, w)))


def compute_loss_MAE(y, tx, w):
    """Calculate the loss using mean absolute error (MAE)"""

    return np.mean(np.abs(y - np.dot(tx, w)))


def compute_gradient(y, tx, w):
    """Calculate the gradient"""

    # basically we have gradient L(w) = -1/N X^Te where e=(y-Xw)
    gradient = -1 / len(y) * np.dot(np.transpose(tx), y - np.dot(tx, w))
    return gradient


def compute_gradient_MAE(y, tx, w):
    """Calculate the gradient using mean absolute error MAE"""

    # on a 1/N sum_1^N |y-x^_ntw|
    # donc le gradient ça devrait être -1/N * X^T sign(y-x^t_nw)
    gradient = -1 / np.shape(tx)[0] * np.dot(np.transpose(tx), np.sign(y - np.dot(tx, w)))
    return gradient


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent algorithm"""

    # Define parameters to store w and loss
    w = initial_w
    w_to_plot = []
    w_to_plot.append(w)

    for n_iter in range(max_iters):
        current_gradient = compute_gradient(y, tx, w)
        if np.linalg.norm(current_gradient) <= 1e-6:
            break
        w = w - gamma / (n_iter + 1) * compute_gradient(y, tx, w)
        w_to_plot.append(w)

    to_plot = np.zeros((max_iters, 1))
    for i in range(max_iters):
        to_plot[i] = compute_loss(y, tx, w_to_plot[i])
    return w,compute_loss(y,tx,w),to_plot
    

    return w, compute_loss(y, tx, w)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent algorithm"""

    n = len(y)
    rnd_sample = np.random.random_integers(0, n - 1, max_iters)
    w_to_plot = []
    w = initial_w

    w_to_plot.append(w)
    for n_iter in range(max_iters):
        subgradient = compute_gradient(y[rnd_sample[n_iter]], tx[rnd_sample[n_iter], :], w)
        if np.linalg.norm(subgradient) <= 1e-6:
            break
        w = w - gamma * subgradient
        w_to_plot.append(w)
    to_plot = np.zeros((max_iters, 1))
    for i in range(max_iters):
        to_plot[i] = compute_loss(y, tx, w_to_plot[i])
    return w,compute_loss(y,tx,w),to_plot

def least_squares(y, tx):
    """Least squares regression using normal equations"""

    x_trans = np.transpose(tx)
    w = np.linalg.solve(np.dot(x_trans, tx), np.dot(x_trans, y))
    return w, compute_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""

    x_trans = np.transpose(tx)

    A = np.dot(x_trans, tx)
    B = np.dot(lambda_ * 2 * len(y), np.identity(x_trans.shape[0]))

    y1 = np.dot(x_trans, y)
    w = np.linalg.solve(A + B, y1)
    
    return w, compute_loss(y, tx, w)


def logistic_regression_S(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using stochastic gradient descent"""
    w = initial_w
    w_to_plot = []

    w_to_plot.append(w)
    rnd_sample = np.random.random_integers(0, len(y) - 1, max_iters)
    for n_iter in range(max_iters):
        s = (sigma(np.dot(tx[rnd_sample[n_iter],:], w)) - y[rnd_sample[n_iter]])
        # nan is an overflow => can be replaced by the function's value at infinity, namely 1
        s = np.where(np.isnan(s), 1, s)

        gradient = tx[rnd_sample[n_iter],:] * s 
        
        if np.linalg.norm(gradient) <= 1e-6 :
            break
        w = w - gamma  * gradient
        w_to_plot.append(w)
    to_plot = np.zeros((max_iters, 1))
    for i in range(max_iters):
        to_plot[i] = compute_loss(y, tx, w_to_plot[i])
    return w,compute_loss(y,tx,w),to_plot
    #return w, compute_loss(y, tx, w)



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent"""
    w = initial_w
    print("shape de transpose tx : " , np.shape(np.transpose(tx)), " and shape of sigma(np.dot(tx,w)) ", np.shape(sigma(np.dot(tx,w))))
    w_to_plot = []
    w_to_plot.append(w)
    for n_iter in range(max_iters):
        s_sigma= sigma(np.dot(tx, w))
        # nan is an overflow => can be replaced by the function's value at infinity, namely 1
        s_sigma= np.where(np.isnan(s_sigma), 1, s_sigma)
        gradient = np.dot(np.transpose(tx), s_sigma-y)
        if np.linalg.norm(gradient) <= 1e-6 :
            break
        w = w - gamma / (n_iter + 1) * gradient
        w_to_plot.append(w)
    to_plot = np.zeros((max_iters, 1))
    for i in range(max_iters):
        to_plot[i] = compute_loss(y, tx, w_to_plot[i])
    return w,compute_loss(y,tx,w),to_plot
    #return w, compute_loss(y, tx, w)

def newton_logistic_regression_s(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    w_to_plot = []

    w_to_plot.append(w)
    rnd_sample = np.random.random_integers(0, len(y) - 1, max_iters)
    for n_iter in range(max_iters):
        s_sigma=sigma(np.dot(tx[rnd_sample[n_iter],:], w))
        # nan is an overflow => can be replaced by the function's value at infinity, namely 1
        
        s = s_sigma - y[rnd_sample[n_iter]]
        # nan is an overflow => can be replaced by the function's value at infinity, namely 1
        s = np.where(np.isnan(s), 1, s)

        gradient = tx[rnd_sample[n_iter],:] * s 
        
        if np.linalg.norm(gradient) <= 1e-6 :
            break
        w = w - gamma/(n_iter + 1)   * np.dot(inverse_hessian(s_sigma,tx[rnd_sample[n_iter],:]), gradient) 
        w_to_plot.append(w)
    to_plot = np.zeros((max_iters, 1))
    for i in range(max_iters):
        to_plot[i] = compute_loss(y, tx, w_to_plot[i])
    return w, compute_loss(y, tx, w),to_plot

   
def sigma(x):

    sigma_output= np.exp(x) / (1 + np.exp(x))
    
    # nan is an overflow => can be replaced by the function's value at infinity, namely 1

    
    return sigma_output

def inverse_hessian(s_sigma,tX):
    S=(s_sigma*(1-s_sigma)) 
    return np.dot(np.transpose(tX),tX)*S
    #S=np.dot(sigma(np.dot(np.tranpose(tX),w)),np.eye(n_row)-



# TODO : check if correct
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent"""
    w = initial_w

    for n_iter in range(max_iters):
        s = sigma(np.dot(tx, w) - y)
        # nan is an overflow => can be replaced by the function's value at infinity, namely 1
        s = np.where(np.isnan(s), 1, s)
        gradient = np.dot(np.transpose(tx), s) + (lambda_ * np.linalg.norm(w)) 
        if np.linalg.norm(gradient) <= 1e-6 :
            print("IT: " , n_iter)
            break
        w = w - (gamma / (n_iter + 1) * gradient)

    return w, compute_loss(y, tx, w)



def sorted_by_variance(tx):
    return np.argsort(np.std(tx, axis=0))


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
        newx[:, current:current + nb_gr_vr * param] = build_poly(tx[:, counter * nb_gr_vr:(counter + 1) * nb_gr_vr],
                                                                 param)
        current = nb_gr_vr * param + current
    newx[:, current:-1] = build_poly(tx[:, (counter + 1) * nb_gr_vr:], params[-1])
    newx[:, -1] = 1
    # print(newx)
    return newx


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=1 up to j=degree"""

    x_flat = np.ndarray.flatten(x)  # put all the features one above the other

    rows, cols = np.indices((x_flat.shape[0], degree))
    res = x_flat[rows]  # creating 'degree' columns

    res[rows, cols] = res[rows, cols] ** (cols + 1)
    res = np.reshape(res, (x.shape[0], -1), 'A')
    return res


def variance_half_max_index(tx):  # TODO rename this function?
    sorted_variance = np.argsort(np.std(tx, axis=0))
    return sorted_variance[len(sorted_variance) // 5:]


def split_data_train_test(y, tx, percentage):
    """split data in percentage to train and 1-percentage to test"""
    len_y = len(y)
    len_test = int(percentage * len_y)
    y_train = y[:len_test]
    y_test = y[len_test:]
    tx_train = tx[:len_test, ]
    tx_test = tx[len_test:, ]
    return y_train, y_test, tx_train, tx_test


def clean_data(tx):
    """Remove lines which contain an element outside [mean - std, mean + std]
    with mean: the mean of the column (feature) and std: the standard deviation of the column (feature)"""
    std_data = np.std(tx, axis=0)
    mean_data = np.mean(tx, axis=0)

    clustered_data = np.concatenate((mean_data - std_data < tx, tx < mean_data + std_data), axis=1)

    return tx[np.all(clustered_data, axis=1), :]


def standardize(x):
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data


# change value which are equal to -999 to 0 TODO (maybe the mean could be an idea too.. certanly a better idea)
def replace_999_data_elem(tx):

    rows,cols=np.shape(tx)
    mean_values_matrix=np.zeros((rows,cols))
    for i in range(cols):
        mean_values_matrix[:,i]=np.mean(tx[tx[:,i]!=-999,i])
    return np.where(tx == -999, mean_values_matrix, tx)



def calculateCovariance(tX, isBinary):
    """
    @param tX: data of features
    @param isBinary: indicates if we want values or only boolean higher than 0.9 or not
    @type isBinary: bool
    @return: correlation matrix
    """
    covariance = np.cov(tX, rowvar=False, bias=True)
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    if isBinary:
        return np.where(correlation >= 0.9, 1, 0)
    correlation[covariance == 0] = 0
    return correlation


def calculateCovariance_y_tX(tX, y):
    """
    calculation unbiased covariance matrix with y, it does not have a lot of impact it would be too easy if there was
    linear link btw y and one  of the features
    """
    y_tX = np.concatenate((np.transpose([y]), tX), axis=1)
    cov_y_tX = calculateCovariance(y_tX, True)
    array_of_corr = np.where(cov_y_tX[:, 0] == 1)
    return array_of_corr


def get_uncorrelated_features(tX):
    """
    Get the features that are uncorrelated, it means it deletes the features that are too much correlated with other
    and discard them
    """
    binary_covariance = calculateCovariance(tX, True)
    n_rows, n_cols = np.shape(binary_covariance)
    columns = np.full((n_rows,), True, dtype=bool)

    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            if binary_covariance[i, j] == 1:
                if columns[j]:
                    columns[j] = False
    return np.where(columns==True)[0]


def test_fct(tX,y,lambda_,i):
    tX_1=replace_999_data_elem(tX)
    features=get_uncorrelated_features(tX_1)
    tX_1=tX_1[:,features]
    #creation of segmentation train and train_test 90% / 10%
    y_train,y_train_test,tx_train,tx_train_test=split_data_train_test(y,tX_1,0.90)
    #calculate of the weights with the train part 
    #tx_train_poly=build_poly(tx_train,5)
    tX_train_poly=build_poly_variance(tx_train,0,i,i,i,i,i)
    weights, loss = ridge_regression(y_train, tX_train_poly, lambda_)
    tX_train_test_poly=build_poly_variance(tx_train_test,0,i,i,i,i,i)
    print("Test: Real  accuracy = ", compute_loss_binary(y_train_test,tX_train_test_poly,weights))
    return compute_loss_binary(y_train_test,tX_train_test_poly,weights)

b=np.array([2,2])
c=np.array([3,2])
