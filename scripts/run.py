import numpy as np
from prediction import *
from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

lambda_ = 2.705e-05
degree = 9
k_fold = 10

jet_indexes = []
for jet in range(4):
    jet_indexes.append(tX[:, 22] == jet)

tX = replace_999_data_elem(tX)

uncorrelated_features = get_uncorrelated_features(tX)
tX = tX[:, uncorrelated_features]

tX_poly = build_poly(tX, degree)


weights_JET = []
acc_jet = []
"""Feature 22 separation"""
for jet in range(4):
    tX_poly_jet = tX_poly[jet_indexes[jet]]
    y_jet = y[jet_indexes[jet]]

    k_indices = build_k_indices(y_jet, k_fold)

    w, loss = ridge_regression(y_jet, tX_poly_jet, lambda_)
    acc_mean, std = cross_validation(w, y_jet, tX_poly_jet, k_fold)

    print("jet :", jet, " Accuracy: ", acc_mean)
    acc_weighted = acc_mean * np.count_nonzero(jet_indexes[jet])

    acc_jet.append(acc_weighted)

acc_jet_mean = np.sum(acc_jet) / len(y)
print("Test: Real  accuracy = ", acc_jet_mean)

DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = '../data/result.csv'
y_pred = predict_y_given_weight(weights_JET, tX_test[:, uncorrelated_features], degree)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
