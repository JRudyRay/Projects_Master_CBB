#!/usr/bin/env python3

'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 2: Decision Trees

Authors: Anja Gumpinger, Bastian Rieck
'''

import numpy as np
import sklearn.datasets
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn import model_selection

# functions
####################################################################
def split_data(X, y, attribute_index, theta):
    # should split X and y into two subsets according to the split defined by the pair
    X1 = [0]*4
    X2 = [0]*4
    y1 = [0]
    y2 = [0]
    for i in range(0,len(y)):
        if X[i, attribute_index] < theta:
            X1 = np.vstack([X1, X[i,:]])
            y1.append(y[i])
        else:
            X2 = np.vstack([X2, X[i,:]])
            y2.append(y[i])
    X1 = np.delete(X1, 0, 0)
    y1 = np.delete(y1, 0, 0)
    X2 = np.delete(X2, 0, 0)  
    y2 = np.delete(y2, 0, 0)          
    return X1,y1,X2,y2
####################################################################
def compute_information_content(y):
    y1 = np.count_nonzero(y == 0)
    y2 = np.count_nonzero(y == 1)
    y3 = np.count_nonzero(y == 2)
    len_y = len(y)
    
    p1 = y1/len_y
    p2 = y2/len_y
    p3 = y3/len_y
    
    if p1 == 0:
        x1 = 0
    else:
        x1 = p1*math.log(p1,2)         
    if p2 == 0:
        x2 = 0
    else:
        x2 = p2*math.log(p2,2)        
    if p3 == 0:
        x3 = 0
    else:
        x3 = p3*math.log(p3,2)
        
    info = -(x1 + x2 + x3)        
    return info
#####################################################################         
def compute_information_a(X, y, attribute_index, theta):
    
    X1,y1,X2,y2 = split_data(X, y, attribute_index, theta)
    
    s1 = (len(X1)/len(X))* compute_information_content(y1)
    s2 = (len(X2)/len(X))* compute_information_content(y2)
    InfoA = s1 + s2
    
    return InfoA
##################################################################### 
# I split the subcalculations into the previous functions   
def compute_information_gain(X, y, attribute_index, theta):
    
    InfoDA = compute_information_a(X, y, attribute_index, theta)
    
    InfoD = compute_information_content(y)
    
    Gain = InfoD - InfoDA
    return Gain
#####################################################################
def class_performance(X, y, n):
    kf = model_selection.KFold(n_splits = 5, shuffle = True) 

    counter = 0
    for train_index, test_index in kf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test   = X[test_index], y[test_index]
        
        clf = DecisionTreeClassifier()        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores = metrics.accuracy_score(y_test, y_pred)
        counter += scores
        
        import_feat = clf.feature_importances_
        x = np.argsort(import_feat)[::-2][:n]
        
    mean_score = (counter/5)*100
    return mean_score, x
###################################################################



if __name__ == '__main__':

    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    num_features = len(set(feature_names))



    print('')
    
    print('Exercise 2.b')
    print('------------')
    print('Split ( sepal length (cm) < 5.5 ): information gain = {0:.2f}'.format(compute_information_gain(X, y, 0, 5.5)))
    print('Split ( sepal width (cm)  < 3.0 ): information gain = {0:.2f}'.format(compute_information_gain(X, y, 1, 3.0)))
    print('Split ( petal length (cm) < 2.0 ): information gain = {0:.2f}'.format(compute_information_gain(X, y, 2, 2.0)))
    print('Split ( petal width (cm)  < 1.0 ): information gain = {0:.2f}'.format(compute_information_gain(X, y, 3, 1.0)))
    print('')
    
    print('Exercise 2.c')
    print('------------')
    print('I would select ( petal length (cm) < 2.0 ) or ( petal width (cm)  < 1.0 ) to be the first split since they both have the same information gain that is highest of all four and the highest information gain indicates lower entropy within the groups, which is what we are looking for.')
    print('')

    ####################################################################
    # Exercise 2.d
    ####################################################################

    # Do _not_ remove this line because you will get different splits
    # which make your results different from the expected ones...
    np.random.seed(42)
    
    # How do I proceed
    # Perform cross-validation using Kfold class from sklearn.model_selection (n_splits = 5, shuffle = True)
    # Do lable prediction with DecisionTreeClassifier from sklearn.tree
    # Train a decision tree on each fold and perform predictions on the corresponding test data
    # use sklearn.metrics.accuracy_Score to evaluate the results and report the MEAN accuracy score in percentage.

    
    
    mean_score, x = class_performance(X, y, 2)
    print('Exercise 2.d')
    print('------------')
    print('The mean accuracy score is {:.2f}'.format(mean_score))
    print('')
    
    print('For the original data, the two most important features are:')
    print('- petal width')
    print('- petal length')
    print('')
    
    X = X[y != 2]
    y = y[y != 2]
    
    mean_score, x = class_performance(X, y, 2)
    
    print('For the reduced data, the most important feature is:')
    print('- petal length')
    print('This means that petal width brought about a perfect split, resulting in a pure node.')
    

    

       
