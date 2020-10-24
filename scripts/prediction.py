import numpy as np
from proj1_helpers import *
from implementations import *


def predict_y_given_weight(weights, tx_test, degree):
    y = np.zeros(tx_test.shape[0])

    for jet in range(4):
        print("shape jet: ", (tx_test[:, 21] == jet).shape, " selected ", np.count_nonzero(tx_test[:, 21] == jet))
        y[tx_test[:, 21] == jet] = predict_labels(weights[jet], build_poly(tx_test[tx_test[:, 21] == jet, :], degree))

    return y
