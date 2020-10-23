import numpy as np
import seaborn as sns  # TODO remove this import: only numpy allowed
import matplotlib.pyplot as plt  # TODO remove this import: only numpy allowed

from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

#DATA_TEST_PATH = '../data/test.csv'
#_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

lambda_ = 2.705e-05
max_iters = 10000
gamma = 1
degree = 9

print("tx shape :", tX.shape)

tX = replace_999_data_elem(tX)

uncorrelated_features = get_uncorrelated_features(tX)
print("uncorr col:", uncorrelated_features)
tX = tX[:, uncorrelated_features]

positive_columns = get_higher_minus_1(tX)
print("possitiv col:", positive_columns)
tX_1 = log_inv(tX[:, positive_columns])
tX = np.concatenate((tX, tX_1), axis=1)

#selected_features = remove_features_with_high_frequencies(tX, 50)
#tX = tX[:, selected_features]


#selected_lines = remove_lines_with_999(tX, 2)
#tX = tX[selected_lines, :]
#y = y[selected_lines]

#indexes_to_keep = clean_data(tX)
#tX = tX[indexes_to_keep, :]
#y = y[indexes_to_keep]

tX_poly = build_poly(standardize(tX), degree)

print("tx shape after preprocessing:", tX.shape)


print('Split data cross validation: 90% train, 10% test tx shape: ')
k_indices = build_k_indices(y, 10)
y_train, y_train_test, tx_train, tx_train_test = cross_validation_data(y, tX_poly, k_indices)


#print('compute weights using logistic regression')
#weights, loss = logistic_regression(y_train, tx_train, np.zeros((tx_train.shape[1]), dtype=float), max_iters, gamma)

print('compute weights using ridge regression')
weights, loss = ridge_regression(y_train, tx_train, lambda_)

print("Test: Real  accuracy = ", compute_accuracy(y_train_test, tx_train_test, weights, is_logistic=True))

#OUTPUT_PATH = '../data/result.csv'
#y_pred = predict_labels(weights, tX_test)
#create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
