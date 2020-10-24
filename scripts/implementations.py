# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:47:51 2020
"""
import numpy as np
from matplotlib import pyplot as plt

from proj1_helpers import *


def compute_accuracy(y_test, tx_test, w_train, is_logistic=False):
    """Calculate accuracy
    
    is_logistic must be True if logistic method is used"""

    predicted_y = np.dot(tx_test, w_train)

    if is_logistic:
        predicted_y = np.where(predicted_y > 0.5, 1, -1)

    predicted_y = np.where(predicted_y > 0, 1, -1)
    sum_y = predicted_y - y_test

    return 1 - np.count_nonzero(sum_y) / len(sum_y)


def compute_loss(y, tx, w, logistic=False):
    """Calculate the loss using mean squared error (MSE)
    
    is_logistic must be True if logistic method is used"""

    if logistic:
        return 0.5 * np.mean(np.square(np.where(y == -1, 0, 1) - np.dot(tx, w)))
    return 0.5 * np.mean(np.square(y - np.dot(tx, w)))


def compute_loss_MAE(y, tx, w, logistic=False):
    """Calculate the loss using mean absolute error (MAE)"""

    if logistic:
        return 0.5 * np.mean(np.square(np.where(y == -1, 0, 1) - np.dot(tx, w)))
    return np.mean(np.abs(y - np.dot(tx, w)))


# ---------------------------------------------------------------------------------
# methods to implement
# ---------------------------------------------------------------------------------

# Compute of gradients

def compute_gradient(y, tx, w):
    """Calculate the gradient"""

    # basically we have gradient L(w) = -1/N X^Te where e=(y-Xw)
    gradient = -1 / np.shape(tx)[0] * np.dot(np.transpose(tx), y - np.dot(tx, w))
    return gradient


def compute_gradient_MAE(y, tx, w):
    """Calculate the gradient using mean absolute error MAE"""

    gradient = -1 / np.shape(tx)[0] * np.dot(np.transpose(tx), np.sign(y - np.dot(tx, w)))
    return gradient


# ############ LEAST SQUARE METHODS ###############################
def least_squares_GD(y, tx, initial_w, max_iters, gamma, toplot=False):
    """Linear regression using gradient descent algorithm"""

    # Define parameters to store w and loss
    w = initial_w
    gradient_to_plot = []

    for n_iter in range(max_iters):
        current_gradient = compute_gradient(y, tx, w)
        if np.linalg.norm(current_gradient) <= 1e-6:
            break
        w = w - gamma / (n_iter + 1) * compute_gradient(y, tx, w)
        gradient_to_plot.append(np.linalg.norm(compute_gradient(y, tx, w)))

    if toplot:
        return w, compute_loss(y, tx, w), gradient_to_plot

    return w, compute_loss(y, tx, w)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, toplot=False):
    """Linear regression using stochastic gradient descent algorithm"""

    n = len(y)
    rnd_sample = np.random.random_integers(0, n - 1, max_iters)
    gradient_to_plot = []
    w = initial_w

    for n_iter in range(max_iters):
        subgradient = compute_gradient(y[rnd_sample[n_iter]], tx[rnd_sample[n_iter], :], w)
        if np.linalg.norm(subgradient) <= 1e-6:
            break
        w = w - gamma * subgradient
        gradient_to_plot.append(np.linalg.norm(subgradient))
    if toplot:
        return w, compute_loss(y, tx, w), gradient_to_plot
    return w, compute_loss(y, tx, w)


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


# ###################### LOGISTIC REGRESSION ###########################

def logistic_regression_S(y, tx, initial_w, max_iters, gamma, toplot=False):
    """Logistic regression using stochastic gradient descent"""

    w = initial_w
    gradient_to_plot = []

    rnd_sample = np.random.random_integers(0, len(y) - 1, max_iters)
    for n_iter in range(max_iters):
        s = (sigma(np.dot(tx[rnd_sample[n_iter], :], w)) - y[rnd_sample[n_iter]])

        gradient = tx[rnd_sample[n_iter], :] * s

        if np.linalg.norm(gradient) <= 1e-6:
            break
        w = w - gamma * gradient
        gradient_to_plot.append(np.linalg.norm(gradient))
    if toplot:
        return w, compute_loss(y, tx, w, True), gradient_to_plot
    return w, compute_loss(y, tx, w, True)


def logistic_regression(y, tx, initial_w, max_iters, gamma: float, toplot=False):
    """Logistic regression using gradient descent"""

    w = initial_w
    gradient_to_plot = []

    for n_iter in range(max_iters):
        s_sigma = sigma(np.dot(tx, w))
        gradient = np.dot(np.transpose(tx), s_sigma - y)
        if np.linalg.norm(gradient) <= 1e-6:
            break
        w = w - gamma / (n_iter + 1) * gradient
        gradient_to_plot.append(np.linalg.norm(gradient))
    if toplot:
        return w, compute_loss(y, tx, w, True), gradient_to_plot
    return w, compute_loss(y, tx, w, True)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent"""

    w = initial_w

    for n_iter in range(max_iters):
        s = sigma(np.dot(tx, w) - y)
        gradient = np.dot(np.transpose(tx), s) + (lambda_ * np.linalg.norm(w))
        if np.linalg.norm(gradient) <= 1e-6:
            break
        w = w - (gamma / (n_iter + 1) * gradient)

    return w, compute_loss(y, tx, w, True)


def newton_logistic_regression_s(y, tx, initial_w, max_iters, gamma, toplot=False):
    """Logistic regression using newton's method with stochastic gradient descent"""

    w = initial_w
    gradient_to_plot = []

    rnd_sample = np.random.random_integers(0, len(y) - 1, max_iters)
    for n_iter in range(max_iters):
        s_sigma = sigma(np.dot(tx[rnd_sample[n_iter], :], w))

        s = s_sigma - y[rnd_sample[n_iter]]
        s = np.where(np.isnan(s), 1, s)

        gradient = tx[rnd_sample[n_iter], :] * s

        if np.linalg.norm(gradient) <= 1e-6:
            break
        w = w - gamma / (n_iter + 1) * np.dot(inverse_hessian(s_sigma, tx[rnd_sample[n_iter], :]), gradient)
        gradient_to_plot.append(np.linalg.norm(gradient))
    if toplot:
        return w, compute_loss(y, tx, w, True), gradient_to_plot
    return w, compute_loss(y, tx, w, True)


def newton_logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using newton's method with gradient descent"""

    w = initial_w

    for n_iter in range(max_iters):
        s_sigma = sigma(np.dot(tx, w))

        gradient = np.dot(np.transpose(tx), s_sigma - y)

        if np.linalg.norm(gradient) <= 1e-6:
            break
        w = w - gamma / (n_iter + 1) * np.dot(inverse_hessian(s_sigma, tx), gradient)
        print("norm gradient: ", np.linalg.norm(gradient))

    return w, compute_loss(y, tx, w, True)


def sigma(x):
    """Apply the sigma function on x"""
    if type(x) == 'numpy.ndarray':
        return_sigma = np.zeros(len(x))
        return_sigma[x > 0] = 1 / (1 + np.exp(-x[x > 0]))
        return_sigma[x <= 0] = np.exp(x[x <= 0]) / (1 + np.exp(x[x <= 0]))
        return return_sigma
    else:
        return np.exp(x) / (1 + np.exp(x))


def inverse_hessian(s_sigma, tX):
    """Calculate the inverse of the hessian"""
    S_flatten = s_sigma * (1 - s_sigma)
    return 1/np.dot(np.transpose(tX) * S_flatten, tX)

# ######## Preprocessing variance


def sorted_by_variance(tx):
    """Sort columns by their variance (increasing order)

    Returns the indexes"""
    return np.argsort(np.std(tx, axis=0))


def build_poly_variance(tx, degree1=0, degree2=3, degree3=5, degree4=4, degree5=5):
    """
    Build polynomial with different degrees depending on their variances respectively

    @param tx:
    @param degree1: degree for the first block of columns
    @param degree2: degree for the second block of columns
    @param degree3: degree for the third block of columns
    @param degree4: degree for the forth block of columns
    @param degree5: degree for the fifth block of columns
    @return: built polynomial array
    """

    n_rows, n_cols = np.shape(tx)

    params = np.array([degree1, degree2, degree3, degree4, degree5])
    nb_gr_vr = n_cols // 5

    # compute total number of columns, we add one for the independent term
    # params[-1] return the last element of params
    n_col_tot = np.sum(params * nb_gr_vr) + 1 + params[-1] * (n_cols % 5)

    newx = np.zeros((n_rows, n_col_tot))
    current = 0

    for counter, param in enumerate(params[:-1]):
        newx[:, current:current + nb_gr_vr * param] = build_poly(tx[:, counter * nb_gr_vr:(counter + 1) * nb_gr_vr],
                                                                 param)
        current = nb_gr_vr * param + current
    newx[:, current:-1] = build_poly(tx[:, (counter + 1) * nb_gr_vr:], params[-1])
    newx[:, -1] = 1
    return newx


def get_higher_minus_1(x):
    """Returns the indexes of columns that contains only value greater than -1"""

    return [i for i in range(np.shape(x)[1]) if (x[:, i] > -1).all()]


def log_inv(x):
    """Calculate the inverse logarithm"""

    return np.log(1 / (1 + x))


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree"""

    x_flat = np.ndarray.flatten(x)  # put all the features one above the other

    rows, cols = np.indices((x_flat.shape[0], degree))
    res = x_flat[rows]  # creating 'degree' columns

    res[rows, cols] = res[rows, cols] ** (cols + 1)
    res = np.reshape(res, (x.shape[0], -1), 'A')
    res = np.concatenate((np.ones((x.shape[0], 1)), res), axis=1)  # add independent term
    return res


def remove_features_with_high_frequencies(tx, percentage):
    """
    Remove all features (columns) that contains element with frequency higher than percentage

    percentage: [0, 1]

    returns an array of bool (false if the feature is removed)
    """

    feature_selected = np.zeros(tx.shape[1], dtype=bool)
    for ind, tx_row in enumerate(np.transpose(tx)):
        unique, counts = np.unique(tx_row, return_counts=True)
        feature_selected[ind] = True if counts[0] / len(tx_row) < percentage else False

    return feature_selected


def remove_lines_with_999(tx, number: int):
    """
    Remove all lines that contain more than 'number' of the value '-999'

    returns an array of bool (false if the line is removed)

    Example:
    # >>> tx = np.array([[1, 2, -999, -999, 5, -999], [1, 2, 3, 4, -999, -999]])
    # >>> remove_lines_with_999(tx, 2)
    # array([False  True])
    """

    lines_selected = np.ones(tx.shape[0], dtype=bool)
    for ind, tx_line in enumerate(tx):
        lines_selected[ind] = True if sum(tx_line == -999) <= number else False

    return lines_selected


def split_data_train_test(y, tx, percentage):
    """
    split data in percentage to train and 1-percentage to test

    percentage: [0, 1]

    Returns y_train, y_test, tx_train, tx_test
    """

    len_y = len(y)
    len_test = int(percentage * len_y)
    y_train = y[:len_test]
    y_test = y[len_test:]
    tx_train = tx[:len_test, ]
    tx_test = tx[len_test:, ]
    return y_train, y_test, tx_train, tx_test


def clean_data(tx):
    """
    Remove lines which contain more than 80% of elements outside [mean - std, mean + std]

    with mean: the mean of the column (feature) and std: the standard deviation of the column (feature)

    Returns an array of bool
    """

    std_data = np.std(tx, axis=0)
    mean_data = np.mean(tx, axis=0)

    clustered_data = np.concatenate((mean_data - std_data < tx, tx < mean_data + std_data), axis=1)

    return np.count_nonzero(clustered_data, axis=1) > clustered_data.shape[1] * 0.8


def standardize(x):
    """
    Center the data and divide by the standard deviation

    Returns std_data, mean, std
    """

    mean = np.mean(x, axis=0)
    centered_data = x - mean
    std = np.std(centered_data, axis=0)
    std_data = centered_data / std

    return std_data, mean, std


def var_mean_of_x(x):
    """
    Calculate the mean and standard deviation of x

    Return mean, std
    """

    return np.mean(x, axis=0), np.std(x, axis=0)


def center_data_given_mean_var(x, var, mean):
    """
    Standardize the data with a given mean and variance

    Returns std_data
    """
    centered_data = x - mean
    std_data = centered_data / var

    return std_data


def replace_999_data_elem(tx, replaced_median=True, values=None):
    """
    Replace all values equal to -999 by the median of the column if replaced_median is True
    
    if replaced_median is False, -999 are replaced by values[column_of_element]

    Returns tx_updated, medians

    note: median is calculated without -999 values
    """

    rows, cols = np.shape(tx)
    median_values_matrix = np.zeros((rows, cols))
    for i in range(cols):
        median_values_matrix[:, i] = np.median(tx[tx[:, i] != -999, i]) if replaced_median else values[i]

    return np.where(tx == -999, median_values_matrix, tx), median_values_matrix[0]


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

    return np.where(columns == True)[0]


def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold"""

    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)


def cross_validation_data(y, tx, k_indices, k=None):
    """
    Returns train data and test data, get k'th subgroup in test, others in train

    Note if k is None, k is chosen random
    """

    if k is None:
        k = np.random.randint(len(k_indices))

    tx_test = tx[k_indices[k]]
    y_test = y[k_indices[k]]

    train_indexes = np.concatenate((k_indices[:k].flatten(), k_indices[k + 1:].flatten()))
    tx_train = tx[train_indexes, :]
    y_train = y[train_indexes]

    return y_train, y_test, tx_train, tx_test


def test_fct(tX, y, lambda_, degree):
    """
    Tests ridge regression using specific lambda and degree for polynomial

    Returns accuracy
    """
    tX_1 = replace_999_data_elem(tX)
    features = get_uncorrelated_features(tX_1)
    tX_1 = tX_1[:, features]
    positive_columns = get_higher_minus_1(tX_1)

    tX_2 = log_inv(tX_1[:, positive_columns])
    tX_1 = np.concatenate((tX_1, tX_2), axis=1)
    tX_1 = standardize(tX_1)

    # creation of segmentation train and train_test 90% / 10%
    y_train, y_train_test, tx_train, tx_train_test = split_data_train_test(y, tX_1, 0.90)

    # calculate of the weights with the train part
    tX_train_poly = build_poly_variance(tx_train, degree, degree, degree, degree, degree)
    weights, loss = ridge_regression(y_train, tX_train_poly, lambda_)
    tX_train_test_poly = build_poly_variance(tx_train_test, degree, degree, degree, degree, degree)
    print("Test: Real  accuracy = ", compute_accuracy(y_train_test, tX_train_test_poly, weights, False))

    return compute_accuracy(y_train_test, tX_train_test_poly, weights, False)
