# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, PairwiseKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')


    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # dropping seasons
    train_df = train_df.drop(['season'], axis = 1)
    test_df = test_df.drop(['season'], axis=1)

    #impute data
    imp = KNNImputer(n_neighbors=2, weights='uniform')
    train_df = imp.fit_transform(train_df)
    test_df = imp.fit_transform(test_df)

    # Dummy initialization of the X_train, X_test and y_train   
    y_train = train_df[:,1]
    X_train = np.delete(train_df, 1, 1)
    X_test = test_df


    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test

    #remove data without labels
    #X_train = X_train[y_train.notnull()]
    #y_train = y_train.dropna()

    # relabel the column 'season' to integers
    #X_train['season'].replace(['spring','summer','autumn','winter'], [0,1,2,3], inplace = True)
    #X_test['season'].replace(['spring','summer','autumn','winter'], [0,1,2,3], inplace = True)

    #Imputation
    #imp = KNNImputer(n_neighbors=2, weights='uniform')
    #X_train = imp.fit_transform(X_train)
    #X_test = imp.fit_transform(X_test)
    #y_train = imp.fit_transform(y_train)


    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, Y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    Y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_pred: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions

    #create validation set from training set
    x_train, x_val, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.27, shuffle=False, random_state=0)
    kernels = [DotProduct(), RBF(), Matern(), RationalQuadratic()]  #, PairwiseKernel(metric='poly', gamma=3.4)
    scores = []

    for kernel in kernels:
        gpr = GaussianProcessRegressor(kernel= kernel, random_state=0)
        gpr.fit(x_train,y_train)

        y_pred = gpr.predict(x_val)
        score = r2_score(y_test,y_pred)
        scores.append(score)

    print(scores)
    best_kernel = kernels[np.argmax(scores)]

    gpr = GaussianProcessRegressor(kernel=best_kernel, random_state=0)
    gpr.fit(X_train,Y_train)

    y_pred = gpr.predict(X_test)


    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

