import numpy as np
from proj1_helpers import *
from implementations import *


def predict_y_given_weight(weights, tx_test, degree):
    """
    Predicts y given the 4 weights and degree

    weights[i] is used to predict y for lines having feature PRI_jet_num == i
    """

    PRI_jet_num_index = 21
    y = np.zeros(tx_test.shape[0])
    for jet in range(4):
        y[tx_test[:, PRI_jet_num_index] == jet] = predict_labels(weights[jet], build_poly(tx_test[tx_test[:, PRI_jet_num_index] == jet, :], degree))

    return y
