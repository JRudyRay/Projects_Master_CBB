'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 1: Logistic Regression

Authors: Anja Gumpinger, Dean Bodenham, Bastian Rieck
'''

#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# functions
def compute_metrics(y_true, y_pred):
    '''
    Computes several quality metrics of the predicted labels and prints
    them to `stdout`.

    :param y_true: true class labels
    :param y_pred: predicted class labels
    '''
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    

    print('TP: {0:d}'.format(tp))
    print('FP: {0:d}'.format(fp))
    print('TN: {0:d}'.format(tn))
    print('FN: {0:d}'.format(fn))
    
    print('Accuracy: {0:.3f}'.format(accuracy_score(y_true, y_pred)))
    '''
    print('Precision: {0:.3f}'.format(precision))
    print('Recall: {0:.3f}'.format(recall))
    print('------------')
    print('Additional values calculated for LDA')
    print('Precision: {0:.3f}'.format(81/(81+48)))
    print('Recall: {0:.3f}'.format(81/(81+28)))
    print('------------')
    '''
# print the results from our analysis


if __name__ == "__main__":

    ###################################################################
    # Your code goes here.
    ###################################################################
    # define file path
    train_file = 'data/diabetes_train.csv'
    test_file = 'data/diabetes_test.csv'

    # read data from file using pandas
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # extract first 7 columns to data matrix X (actually, a numpy ndarray)
    X_train = df_train.iloc[:, 0:7].values
    X_test = df_test.iloc[:, 0:7].values

    # extract 8th column (labels) to numpy array
    Y_train = df_train.iloc[:, 7].values
    Y_test = df_test.iloc[:, 7].values

    # i) Standardization aka scale
    stand = StandardScaler()
    X_train_scal = stand.fit_transform(X_train)
    X_test_scal = stand.transform(X_test)

    # ii) train the model
    reg = LogisticRegression(C=1.0, solver="lbfgs")
    mod = reg.fit(X_train_scal, Y_train)

    # iii) compute Y_pred
    Y_pred = mod.predict(X_test_scal)
    
    
    
    print('Exercise 1.a')
    print('------------')
    compute_metrics(Y_test, Y_pred)
    print('')
    
    print('Exercise 1.b')
    print('------------')
    print('For the diabetes dataset I would choose LDA, because it has a higher count of true potsitives and a lower count of false negatives than LR.')
    print('')

    print('Exercise 1.c')
    print('------------')
    print('For another dataset, I would choose Logistic Regression, because LDA makes more assumptions than LR and therefore LR is more robust in the case that those assumptions are violated.')
    print('')

    print('Exercise 1.d')
    print('------------')
    '''
    print(list(df_train.columns[0:7]))
    print(mod.coef_)
    print(np.exp(mod.coef_))
    '''
    print('The two attributes which appear to contribute most to the prediction are glu and ped.')
    print('')
    print('The coefficient for npreg is 0.33. Calculating the exponential function results in 1.40, which amounts to an increase in diabetes risk of 40 percent per additional pregnancy.')


