import numpy as np
import seaborn as sns  # TODO remove this import: only numpy allowed
import matplotlib.pyplot as plt  # TODO remove this import: only numpy allowed

from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

lambda_ = 2.705e-05
max_iters = 10000
gamma = 1
degree = 9
k_fold = 10

print("tx shape :", tX.shape)
# jet_selected = tX[:, 22] == 3
# tX = tX[jet_selected]
# y = y[jet_selected]

tX = replace_999_data_elem(tX)

uncorrelated_features = get_uncorrelated_features(tX)
tX = tX[:, uncorrelated_features]

positive_columns = get_higher_minus_1(tX)
tX_1 = log_inv(tX[:, positive_columns])
tX = np.concatenate((tX, tX_1), axis=1)

#selected_features = remove_features_with_high_frequencies(tX, 1)
#tX = tX[:, selected_features]


#selected_lines = remove_lines_with_999(tX, 2)
#tX = tX[selected_lines, :]
#y = y[selected_lines]

#indexes_to_keep = clean_data(tX)
#tX = tX[indexes_to_keep, :]
#y = y[indexes_to_keep]

tX_poly = build_poly(standardize(tX), degree)

print("tx shape after preprocessing:", tX_poly.shape)


print('Split data cross validation: 90% train, 10% test tx shape: ')
k_indices = build_k_indices(y, k_fold)

weights_cross = []
acc_cross = []
for k in range(k_fold):
    y_train, y_train_test, tx_train, tx_train_test = cross_validation_data(y, tX_poly, k_indices, k)

    w, loss = ridge_regression(y_train, tx_train, lambda_)
    weights_cross.append(w)
    acc_cross.append(compute_accuracy(y_train_test, tx_train_test, w))

weights_mean = np.mean(weights_cross, axis=0)
acc_mean = np.mean(acc_cross)

print("Test: Real  accuracy = ", acc_mean)


#DATA_TEST_PATH = '../data/test.csv'
#_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#OUTPUT_PATH = '../data/result.csv'
#y_pred = predict_labels(weights_mean, tX_test[:, uncorrelated_features])
#create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
