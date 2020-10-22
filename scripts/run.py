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
max_iters = 200
gamma = 1e-10

selected_features = remove_features_with_high_frequencies(tX, 10)
selected_lines = remove_lines_with_999(tX[:, selected_features], 2)

print('Split data cross validation: 80% train, 20% test')
k_indices = build_k_indices(y, 5)
y_train, y_train_test, tx_train, tx_train_test = cross_validation_data(y[selected_lines],
                                                                       tX[:, selected_features][selected_lines, :],
                                                                       k_indices)

print('compute weights using newton logistic regression')
weights, loss = newton_logistic_regression(y_train, tx_train, np.zeros((tx_train.shape[1]), dtype=float), max_iters, gamma)

print("Test: Real  accuracy = ", compute_loss_binary(y_train_test, tx_train_test, weights))

OUTPUT_PATH = '../data/result.csv'
y_pred = predict_labels(weights, tX_test[:, selected_features])
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
