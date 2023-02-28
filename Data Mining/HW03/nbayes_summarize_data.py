"""Homework 1: Distance functions on vectors.

Homework 3: Part 1: Naive Bayes
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
    "--outdir",
    required=True,
    help="Path to directory where output_knn.txt will be created"
)

args = parser.parse_args()

# Set the paths
traindir = args.traindir

os.makedirs(args.outdir, exist_ok=True)


# importing the data
dtrain = pd.read_csv("{}/{}".format(args.traindir,'tumor_info.txt'), sep='\t', header = None)

# Create the output file class 2
try:
    file_name = "{}/output_summary_class_2.txt".format(args.outdir) ######## REPLACE THE LABLE AND ADD ANOTHER OUTPUT FILE
    f_out = open(file_name, 'w')
except IOError:
    print("Output file {} cannot be created".format(file_name))
    sys.exit(1)


# Write header for output file
f_out.write('{}\t{}\t{}\t{}\t{}\n'.format(
    'Value',
    'clump',
    'uniformity',
    'marginal',
    'mitoses'))

beg = dtrain[dtrain[4] == 2]
mal = dtrain[dtrain[4] == 4]

p_beg = [0] * (len(beg.axes[1])-1)
p_mal = [0] * (len(mal.axes[1])-1)

rows_beg = len(beg.axes[0])-1
rows_mal = len(mal.axes[0])-1

beg = beg.to_numpy()
mal = mal.to_numpy()

nrows, ncols = beg.shape
ncols = ncols -1


for value in range(1,11): # iterate through values
    for j in range(ncols): # iterate through columns
        count = 0 # initialize counter
        count_nan = 0 # initialize NA counter
        for i in range(nrows): # iterate through rows
        
            if beg[i,j] == value:
                count +=1
            if math.isnan(beg[i,j]):
                count_nan += 1    
        p_beg[j] = np.round(count/(nrows-count_nan),3)
    
    
    # Save the output
    f_out.write(
        '{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
            value,
            p_beg[0],
            p_beg[1],
            p_beg[2],
            p_beg[3]))
    
f_out.close()


################################################################

# importing the data
dtrain = pd.read_csv("{}/{}".format(args.traindir,'tumor_info.txt'), sep='\t', header = None)

# Create the output file class 2
try:
    file_name = "{}/output_summary_class_4.txt".format(args.outdir) ######## REPLACE THE LABLE AND ADD ANOTHER OUTPUT FILE
    f_out = open(file_name, 'w')
except IOError:
    print("Output file {} cannot be created".format(file_name))
    sys.exit(1)


# Write header for output file
f_out.write('{}\t{}\t{}\t{}\t{}\n'.format(
    'Value',
    'clump',
    'uniformity',
    'marginal',
    'mitoses'))

mal = dtrain[dtrain[4] == 4]

p_mal = [0] * (len(mal.axes[1])-1)

mal = mal.to_numpy()

nrows, ncols = mal.shape
ncols = ncols -1

for value in range(1,11): # iterate through values
    for j in range(ncols): # iterate through columns
        count = 0 # initialize counter
        count_nan = 0
        for i in range(nrows): # iterate through rows
        
            if mal[i,j] == value:
                count +=1
            if math.isnan(mal[i,j]):
                count_nan += 1
                
        p_mal[j] = np.round(count/(nrows-count_nan),3)
    
    
    # Save the output
    f_out.write(
        '{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
            value,
            p_mal[0],
            p_mal[1],
            p_mal[2],
            p_mal[3]))
    
f_out.close()


#####################################################################




