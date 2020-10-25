# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:25:09 2020

@author: jeang
"""

import numpy as np
from prediction import *
from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
lambda_ = 1e-9
degree = 8
k_fold = 20

positive_columns=np.intersect1d(get_higher_minus_1(tX[:,uncorrelated_features]),get_higher_minus_1(tX_test[:,uncorrelated_features]))
jet_indexes = []
for jet in range(4):
    jet_indexes.append(tX[:, 22] == jet)  # PRI_jet_num is the 22 feature

tX, median_values_matrix = replace_999_data_elem(tX)

uncorrelated_features = get_uncorrelated_features(tX)
tX = tX[:, uncorrelated_features]
print(np.shape(tX_test[:,uncorrelated_features]))
tX_2=log_inv(tX[:,positive_columns])
tX=np.concatenate((tX,tX_2),axis=1)

tX, mean, var =  standardize(tX)

tX_poly = build_poly(tX, degree)
"""Feature PRI_jet_num separation"""
weights_JET = []
acc_jet = []
for jet in range(4):  # PRI_jet_num contains 4 different values
    tX_poly_jet = tX_poly[jet_indexes[jet]]
    y_jet = y[jet_indexes[jet]]

    k_indices = build_k_indices(y_jet, k_fold)

    """cross validation"""
    weights_cross = []
    acc_cross = []
    print(k_fold)
    for k in range(k_fold):
        y_train, y_train_test, tx_train, tx_train_test = cross_validation_data(y_jet, tX_poly_jet, k_indices, k)
        w, loss = ridge_regression(y_train, tx_train, lambda_)
        weights_cross.append(w)
        acc_cross.append(compute_accuracy(y_train_test, tx_train_test, w, False))

    weights_mean = np.mean(weights_cross, axis=0)
    acc_mean = np.mean(acc_cross)
    acc_weighted = acc_mean * np.count_nonzero(jet_indexes[jet])

    weights_JET.append(weights_mean)
    acc_jet.append(acc_weighted)

acc_jet_mean = np.sum(acc_jet) / len(y)
print("Accuracy = ", acc_jet_mean)



OUTPUT_PATH = '../data/result.csv'
tX_test,median_test = (replace_999_data_elem(tX_test, False, median_values_matrix))
tX_test = tX_test[:,uncorrelated_features]

tX_log_test = log_inv(tX_test[:,positive_columns])
tX_test = np.concatenate( ( tX_test , tX_log_test ) , axis = 1 )
tX_test = center_data_given_mean_var(tX_test, var, mean)
print(np.shape(tX_test))
y_pred = predict_y_given_weight(weights_JET, tX_test, degree)

#create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
