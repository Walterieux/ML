# Machine Learning CS-433 - Project 1

All python files are in folder `script`, `train.csv` and `test.csv` are in folder `data`. Report in LaTeX and PDF is located in `LaTex` folder.

- **`implementations.py`** contains the following functions:
	
	- `compute_accuracy` Calculates accuracy of w with y_test and tx_test

	- `compute_loss` Calculates the loss using mean squared error (MSE)

	- `compute_loss_MAE` Calculates the loss using mean absolute error (MAE)

	- `compute_gradient` Calculates the gradient using mean absolute error MAE

	- `least_squares_GD` Linear regression using gradient descent algorithm

	- `least_squares_SGD` Linear regression using stochastic gradient descent algorithm

	- `least_squares` Least squares regression using normal equations

	- `ridge_regression` Ridge regression using normal equations

	- `logistic_regression_S` Logistic regression using stochastic gradient descent

	- `logistic_regression` Logistic regression using gradient descent

	- `reg_logistic_regression` Regularized logistic regression using gradient descent
 
	- `newton_logistic_regression_s` Logistic regression using newton's method with stochastic gradient descent

	- `newton_logistic_regression` Logistic regression using newton's method with gradient descent

	Preprocessing functions:	
	
	- `build_poly_variance` Builds polynomial with different degrees depending on their variances respectively

	- `build_poly` polynomial basis functions for input data x, for j=0 up to j=degree

	- `remove_features_with_high_frequencies` Removes all features (columns) that contains element with frequency higher than percentage

	- `remove_lines_with_999` Removes all lines that contain more than 'number' of the value '-999'

	- `split_data_train_test` splits data in percentage to train and 1-percentage to test

	- `clean_data` Removes lines which contain more than 80% of elements outside [mean - std, mean + std]

	- `standardize` Center the data and divide by the standard deviation
 
	- `replace_999_data_elem` Replaces all values equal to -999 by the median of the column or a defined value passed in argument

	- `get_uncorrelated_features` Gets the features that are uncorrelated

- **`prediction.py`** contains the following functions:

	- `predict_y_given_weight` Predicts values given weight and degree

- **`run.py`** :
	Computes `result.csv` we submitted on aicrowd using `train.csv` for training. `result.csv` is stored in data/. 
