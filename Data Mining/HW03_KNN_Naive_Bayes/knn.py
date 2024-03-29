"""Homework 1: Distance functions on vectors.

Homework 3: Part 1: k-NN algorithm
Course    : Data Mining (636-0018-00L)

"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import math



# EUCLIDEAN DISTANCE
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



# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(
    description="Compute distance functions on time-series"
)
parser.add_argument(
    "--traindir",
    required=True,
    help="Path to the directory where the training data are stored."
)
parser.add_argument(
    "--testdir",
    required=True,
    help="Path to the directory with the test data"
)
parser.add_argument(
    "--outdir",
    required=True,
    help="Path to directory where output_knn.txt will be created"
)
parser.add_argument(
    "--mink",
    required=True,
    help="The minimum value of k on which k-NN algorithm will be invoked"
)
parser.add_argument(
    "--maxk",
    required=True,
    help="The maximum value of k on which k-NN algorithm will be invoked"
)


args = parser.parse_args()

# Set the paths
testdir = args.testdir
traindir = args.traindir
mink = args.mink
maxk = args.maxk

os.makedirs(args.outdir, exist_ok=True)


# importing the data
dtest = pd.read_table("{}/{}".format(args.testdir,'matrix_mirna_input.txt'))
ptest = pd.read_table("{}/{}".format(args.testdir,'phenotype.txt'))
dtest = dtest.to_numpy()
ptest = ptest.to_numpy()

dtrain = pd.read_table("{}/{}".format(args.traindir,'matrix_mirna_input.txt'))
ptrain = pd.read_table("{}/{}".format(args.traindir,'phenotype.txt'))
dtrain = dtrain.to_numpy()
ptrain = ptrain.to_numpy()




# Create the output file
try:
    file_name = "{}/output_knn.txt".format(args.outdir)
    f_out = open(file_name, 'w')
except IOError:
    print("Output file {} cannot be created".format(file_name))
    sys.exit(1)


# Write header for output file
f_out.write('{}\t{}\t{}\t{}\n'.format(
    'Value of k',
    'Accuracy',
    'Precision',
    'Recall'))

kmin = int(args.mink)
kmax = int(args.maxk)

## KNN algorithm
for k in range(kmin,kmax+1):
    y_preds = [0] * len(ptest)
    counter = 0
    # compare every dtest vector to every dtrain vector
    dist_vec = [0] * len(dtrain)
    for i in range(len(dtest)): #
        for j in range(len(dtrain)):
            v1 = dtest[i,1:]
            v2 = dtrain[j,1:]
            dist_vec[j] = euclidean_dist(v1, v2)
    
        dist_vec = np.array(dist_vec) # convert to numpy array
        # get indexes of minimal values
        idx = np.argpartition(dist_vec, k)[:k]
        '''
        pos_vec = [0] * k
        for i in range(k):    
            pos_vec[i] = np.where(idx == i)[0][0]
        '''   
        labels = ptrain[idx,1]
        npos = 0
        nneg = 0
        for l in range(k):
            if labels[l] == '+':
                npos += 1
            if labels[l] == '-':
                nneg += 1
        
        if npos > nneg:
            y_preds[counter] = 1
        if nneg > npos:
            y_preds[counter] = -1
        else:
            nk = k-1        
            idx = np.argpartition(dist_vec, nk)[:nk]
            
            labels = ptrain[idx,1]
            npos = 0
            nneg = 0
            for l in range(nk):
                if labels[l] == '+':
                    npos += 1
                if labels[l] == '-':
                    nneg += 1
            if npos > nneg:
                y_preds[counter] = 1
            if nneg > npos:
                y_preds[counter] = -1
            
        counter += 1
    
    # creating y_trues array
    y_trues = [0] * len(ptest)
    for p in range(len(ptest)):
        if ptest[p,1] == '+':
            y_trues[p] = 1
        if ptest[p,1] == '-':
            y_trues[p] = -1


    acc_toy     = np.round(compute_accuracy(y_trues, y_preds),2)
    prec_toy    = np.round(compute_precision(y_trues, y_preds),2)
    rec_toy     = np.round(compute_recall(y_trues, y_preds),2)
    TP_rate_toy = np.round(compute_tp_rate(y_trues, y_preds),2)
    FP_rate_toy = np.round(compute_fp_rate(y_trues, y_preds),2)
    
    
    # Save the output
    f_out.write(
        '{}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(
            k,
            acc_toy,
            prec_toy,
            rec_toy))
    
f_out.close()



