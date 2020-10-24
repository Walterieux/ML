import numpy as np
from prediction import *
from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

lambda_ = 1.1e-4
degree = 7
k_fold = 20

jet_indexes = []
for jet in range(4):
    jet_indexes.append(tX[:, 22] == jet)  # PRI_jet_num is the 22 feature

tX, median_values_matrix = replace_999_data_elem(tX)

uncorrelated_features = get_uncorrelated_features(tX)
tX = tX[:, uncorrelated_features]

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

DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = '../data/result.csv'
y_pred = predict_y_given_weight(weights_JET, (replace_999_data_elem(tX_test, False, median_values_matrix)[0])[:, uncorrelated_features], degree)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
