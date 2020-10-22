import numpy as np
import seaborn as sns  # TODO remove this import: only numpy allowed
import matplotlib.pyplot as plt  # TODO remove this import: only numpy allowed

from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

lambda_ = 0.00
max_iters = 10000
gamma = 100

print("tx shape :", tX.shape)
indexes_to_keep = clean_data(tX)
tX = tX[indexes_to_keep, :]
y = y[indexes_to_keep]
print("tx shape :", tX.shape)

selected_features = remove_features_with_high_frequencies(tX, 10)
tX = tX[:, selected_features]

selected_lines = remove_lines_with_999(tX, 3)
tX = tX[selected_lines, :]
y = y[selected_lines]

tX = replace_999_data_elem(tX)

tX = build_poly(tX, 5)
print("tX shape after precomputing: ", tX.shape)

print('Split data cross validation: 80% train, 20% test tx shape: ')
k_indices = build_k_indices(y, 5)
y_train, y_train_test, tx_train, tx_train_test = cross_validation_data(y, standardize(tX), k_indices)


print('compute weights using logistic regression')
weights, loss = logistic_regression(y_train, tx_train, np.zeros((tx_train.shape[1]), dtype=float), max_iters, gamma)

print("Test: Real  accuracy = ", compute_accuracy(y_train_test, tx_train_test, weights, is_logistic=True))

OUTPUT_PATH = '../data/result.csv'
y_pred = predict_labels(weights, tX_test[:, selected_features])
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
