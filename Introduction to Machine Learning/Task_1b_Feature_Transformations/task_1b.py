# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    def linear(X):
        return X
    def quadratic(X):
        return X**2
    def exponential(X):
        return np.exp(X)
    def cosine(X):
        return np.cos(X)
    
    functions = [linear, quadratic, exponential, cosine]
    
    X_transformed = np.zeros((700, 21))
    n_x = np.shape(X)[1]
    
    count = 0
    
    for i in range(0, n_x * 4, 5):
        X_transformed[:,i:i+5] = functions[count](X)
        count += 1
        
    X_transformed[:,-1] = 1
    
    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y, lam):
    """
    This function receives training data points, then fits the ridge regression on this data
    with regularization hyperparameter lambda. The weights w of the fitted ridge regression
    are returned. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 13 features
    y: array of floats, dim = (700,), input labels)
    lam: float. lambda parameter, used in regularization term

    Returns
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    """ 
    
    n_features = np.shape(X)[1]
    w = np.zeros((n_features,))
    w = (np.linalg.inv((np.transpose(X) @ X) + lam * np.identity(n_features))) @ np.transpose(X) @ y
    assert w.shape == (n_features,)
    return w

def calculate_RMSE(w, X, y):
    """This function takes test data points (X and y), and computes the empirical RMSE of 
    predicting y from X using a linear model with weights w. 

    Parameters
    ----------
    w: array of floats: dim = (21,), optimal parameters of ridge regression 
    X: matrix of floats, dim = (700,21), inputs with 13 features
    y: array of floats, dim = (700,), input labels

    Returns
    ----------
    RMSE: float: dim = 1, RMSE value
    """
    RMSE = 0
    y_pred = X @ w
    n = np.shape(y)[0]
    RMSE = np.sqrt((1/n) * sum((y-y_pred)**2))
    assert np.isscalar(RMSE)
    return RMSE

def average_LR_RMSE(X, y, lambdas, n_folds):
    """
    Main cross-validation loop, implementing 10-fold CV. In every iteration (for every train-test split), the RMSE for every lambda is calculated, 
    and then averaged over iterations.
    
    Parameters
    ---------- 
    X: matrix of floats, dim = (700, 21), inputs with 13 features
    y: array of floats, dim = (700, ), input labels
    lambdas: list of floats, len = 12, values of lambda for which ridge regression is fitted and RMSE estimated
    n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV
    
    Returns
    ----------
    avg_RMSE: array of floats: dim = (12,), average RMSE value for every lambda
    """
    n_lambdas = len(lambdas)
    RMSE_mat = np.zeros((n_folds, n_lambdas))

    kf = KFold(n_splits = n_folds)
  
    row = 0
    for train, test in kf.split(X):
        
        column = 0
        
        for lambda_value in lambdas:

            w = fit(X[train], y[train], lambda_value)
            RMSE = calculate_RMSE(w, X[test], y[test])
            RMSE_mat[row,column] = RMSE
            
            column += 1

        row += 1

    avg_RMSE = np.mean(RMSE_mat, axis=0)
    assert avg_RMSE.shape == (n_lambdas,)
    return avg_RMSE


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    lambdas = [0.001, 0.01, 0.1, 1, 2, 5, 10, 15, 20, 40, 41, 42, 43, 44, 45, 50, 55, 60, 70, 100]
    n_folds = 10
    # The function retrieving optimal LR parameters
    X_transformed = transform_data(X)
    avg_RMSE = average_LR_RMSE(X_transformed, y, lambdas, n_folds)
    best_lambda = lambdas[np.argmin(avg_RMSE)]
    w = fit(X_transformed, y, best_lambda)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
