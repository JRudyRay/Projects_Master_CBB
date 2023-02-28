# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 02:06:03 2022

@author: johno
"""


import numpy as np
import pandas as pd
import math
import statistics as stats
#f = open("matrix_mirna_input.txt", "r")
#print(f.read())

# EUCLIDEAN distance function
def euclidean_dist(v1, v2):
    return float(math.sqrt(np.sum((v1 - v2)**2)))

# CONFUSION matrix function
def confusion_matrix(y_trues, y_preds):
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for i in range(len(y_trues)):
        if y_trues[i] == 1:
            if y_preds[i] == 1:
                tp += 1
            else:
                fn += 1
        if y_trues[i] == -1:
            if y_preds[i] == -1:
                tn += 1
            else:
                fp += 1
                
    return np.array([[tp, fp], [fn, tn]], dtype = float)

TP, FP, FN, TN = (0, 0), (0, 1), (1, 0), (1, 1)

def compute_accuracy(y_true, y_pred):
    mat_counts = confusion_matrix(y_true, y_pred) # call the confusion matrix with the true and predicted labels as input
    return (mat_counts[TP] + mat_counts[TN]) / mat_counts.sum() # sum of TP and TN divided by all the predictions performed
# Note: Instead of mat_counts[TP] --> you could also type mat_counts[0,0] --> hard coding is less efficient 
#       and more prone to error.

def compute_precision(y_true, y_pred):
    mat_counts = confusion_matrix(y_true, y_pred)
    return mat_counts[TP] / (mat_counts[TP] + mat_counts[FP])

def compute_recall(y_true, y_pred):
    mat_counts = confusion_matrix(y_true, y_pred)
    return mat_counts[TP] / (mat_counts[TP] + mat_counts[FN])

def compute_tp_rate(y_true, y_pred):
    return compute_recall(y_true, y_pred)

def compute_fp_rate(y_true, y_pred):
    mat_counts = confusion_matrix(y_true, y_pred)
    return mat_counts[FP] / (mat_counts[TN] + mat_counts[FP])







# importing the data
dtest = pd.read_table('C:/Users/johno/Documents/University/ETH HS22/Data Mining I/HW/HW03/data/part1/test/matrix_mirna_input.txt')
ptest = pd.read_table('C:/Users/johno/Documents/University/ETH HS22/Data Mining I/HW/HW03/data/part1/test/phenotype.txt')
dtest = dtest.to_numpy()
ptest = ptest.to_numpy()

dtrain = pd.read_table('C:/Users/johno/Documents/University/ETH HS22/Data Mining I/HW/HW03/data/part1/train/matrix_mirna_input.txt')
ptrain = pd.read_table('C:/Users/johno/Documents/University/ETH HS22/Data Mining I/HW/HW03/data/part1/train/phenotype.txt')
dtrain = dtrain.to_numpy()
ptrain = ptrain.to_numpy()



# creating y_trues array
y_trues = [0] * len(ptest)
for i in range(len(ptest)):
    if ptest[i,1] == '+':
        y_trues[i] = 1
    if ptest[i,1] == '-':
        y_trues[i] = -1



k = 3 # change this to the input argument from CL
y_preds = [0] * len(dtest)
counter = 0 # for y_preds vector position

# compare every dtest vector to every dtrain vector
dist_vec = [0] * len(dtrain)
dist_vec = np.array(dist_vec)

for i in range(0,len(dtest)): #
    for j in range(0,len(dtrain)):
        v1 = dtest[i,1:]
        v2 = dtrain[j,1:]
        dist_vec[j] = euclidean_dist(v1, v2)
        print(dist_vec[j])
    dist_vec = np.array(dist_vec) # convert to numpy array
    # get indexes of minimal values
    idx = np.argpartition(dist_vec, k)

    pos_vec = [0] * k
    pos_vec = np.array(pos_vec)
    for i in range(k):    
        pos_vec[i] = np.where(idx == i)[0]
   
    print(dist_vec[pos_vec])
    labels = ptrain[pos_vec,1]
  
    
    npos = 0
    nneg = 0
    for i in range(len(labels)):
        if labels[i] == '+':
            npos += 1
        else:
            nneg += 1
    
    if npos > nneg:
        y_preds[counter] = 1
    if nneg > npos:
        y_preds[counter] = -1

        
    counter += 1



# creating y_trues array
y_trues = [0] * len(ptest)
for i in range(len(ptest)):
    if ptest[i,1] == '+':
        y_trues[i] = 1
    if ptest[i,1] == '-':
        y_trues[i] = -1

#print(len(dtest))
#print(len(dtrain))


## Next I want to iterate through y_preds and ptest and see if they are the same

x = confusion_matrix(y_trues, y_preds)
print(x)



############################# MAIN ################################
# Calculate the performance measures on the small toy-example
acc_toy     = np.round(compute_accuracy(y_trues, y_preds),3)
prec_toy    = np.round(compute_precision(y_trues, y_preds),3)
rec_toy     = np.round(compute_recall(y_trues, y_preds),3)
TP_rate_toy = np.round(compute_tp_rate(y_trues, y_preds),3)
FP_rate_toy = np.round(compute_fp_rate(y_trues, y_preds),3)

# Print results
print('Accuracy = (TP + TN)/(TP + TN + FP + FN) = ' + str(np.round(acc_toy, 3)))
print('Precision = TP/(TP + FP) = ' + str(np.round(prec_toy, 3)))
print('Recall = TP/(TP + FN) = ' + str(np.round(rec_toy, 3)))
print('TP rate = TP/(TP + FN) = ' + str(np.round(TP_rate_toy, 3)))
print('FP rate = FP/(TN + FP) = ' + str(np.round(FP_rate_toy, 3)))


