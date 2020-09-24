# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np 
def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    For this we will calculate MSE
    """
    total= 0.5 * np.mean(np.squared(y-np.dot(np.tranpose(tx),w)))
    return total 

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    # ***************************************************
