## Ridge Regression

The objective of the Task 1a was to implement the Ridge Regression algorithm. The aim is finding the optimal values of the hyperparameter lambda.
To do this we applied 10-fold cross-validation to calculate the Root Mean Squared Error (RMSE) for all the different values of lambda we were given.

In the first function fit(X, y, lam) we fit the ridge regression model on the training data X, labels y and the hyperparameter lambda.
What this does is calculate the best weights, w, for the fitted ridge regression model. Here we used the closed-form solution to the problem. w is what is returned form the function.

In the second function, calculate_RMSE(w, X, y), we calculated the root mean squared error (RMSE) of the predictions of y which we obtain using the data X and the previously calculated weights w.

In the third function average_LR_RMSE(X, y, lambdas, n_folds) we performed 10-fold cross validation. Here we split the data into 10 folds and then train the regression on 9 of those folds and use the single remaining fold for testing.
Here we call upon the fit(X, y, lam) function and then pipe the returned w into the calculate_RMSE(w, X, y) function, which returns the RMSE which we then stored in a matrix RMSE_mat.
We did this over all lambdas. After all loops are done we then calculated the mean of the rows of the RMSE_mat matrix, so the average of the lambdas. The average values are then returned.
