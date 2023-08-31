## Kernelized Regression

In the task 2 we implemented kernelized regression, where we test multiple kernels and select the one that gives the best prediction based on the R2 score.
In addition we performed imputation using the KNN strategy.

In the function data_loading we import the data, drop the columns 'seasons', impute on the data using KNNImputer and then split up the data as necessary.

We then build the model in modeling_and_prediction, where we first test multiple kernels to use for the kernelized regression and then use the one that returns the best R2 score.
We then fit the the model with the GaussianProcessRegressor function using the best kernel we found.
