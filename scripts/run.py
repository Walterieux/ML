import numpy as np
import seaborn as sns  # TODO remove this import: only numpy allowed
import matplotlib.pyplot as plt  # TODO remove this import: only numpy allowed

from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = '../data/test.csv'
_, tX_test1, ids_test = load_csv_data(DATA_TEST_PATH)  # TODO rename tX_test1!

lambda_ = 0.00

# tX,tX_test=replace_999_data_elem(tX,tX_test)
print("shape tx :", np.shape(tX), " and shape tx_test", np.shape(tX_test1))

# tX_1,tX_test=replace_999_data_elem(tX,tX_test1)
tX_1 = replace_999_data_elem(tX)
features = get_uncorellated_features(tX_1)
tX_test = replace_999_data_elem(tX_test1)
print(features)
#### on prend que les features importants pas ceux qui sont correlés à plus de 90% avec les autres
tX_1 = tX_1[:, features]

tX_test = tX_test1[:, features]
# index_variance=sorted_by_variance(tX)
# tX=tX[:,index_variance]
# tX_test=tX_test[:,index_variance]
# print ('Covariance matrix:\n', ACov)
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(10, 10)

ax0 = plt.subplot(2, 2, 1)
plt.title("Correlation matrix of different features")
# Choosing the colors
cmap = sns.color_palette("GnBu", 10)
sns.heatmap(calculateCovariance(replace_999_data_elem(tX), 0), cmap=cmap, vmin=0)
# plt.imshow(calculateCovariance(tX,1), cmap='Greys',  interpolation='nearest')

ax1 = plt.subplot(2, 2, 2)
plt.title("Correlation matrix with black box when values exceed 90%")
# data can include the colors
plt.imshow(calculateCovariance(replace_999_data_elem(tX), True), cmap='Greys', interpolation='nearest')

# Remove the top and right axes from the data plot
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
print("shape of tX : ", np.shape(tX_1), "  and shape of tX_test", np.shape(tX_test))

# creation of segmentation train and train_test 90% / 10%
y_train, y_train_test, tx_train, tx_train_test = data_train_test(y, tX_1, 0.90)

# calculate of the weights with the train part
# tx_train_poly=build_poly(tx_train,5)
tX_train_poly = build_poly_variance(tx_train, 0, 8, 8, 8, 8, 8)
print("train poly :", np.shape(tX_train_poly))
weights, loss = ridge_regression(y_train, tX_train_poly, lambda_)
# tX_train_test_poly=build_poly(tx_train_test,5)
tX_train_test_poly = build_poly_variance(tx_train_test, 0, 8, 8, 8, 8, 8)
print("train test :", np.shape(tX_train_test_poly))
print("Test: Loss = ", compute_loss(y_train_test, tX_train_test_poly, weights))

tX_test_poly = build_poly_variance(tX_test, 0, 8, 8, 8, 8, 8)

OUTPUT_PATH = '../data/result.csv'
y_pred = predict_labels(weights, tX_test_poly)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
