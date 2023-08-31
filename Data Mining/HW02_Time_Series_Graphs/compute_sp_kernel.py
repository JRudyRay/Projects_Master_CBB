# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:05:10 2022

@author: johno
"""

from shortest_path_kernel import floyd_warshall
from shortest_path_kernel import sp_kernel
import os
import sys
import argparse
import numpy as np
import scipy.io

mat = scipy.io.loadmat('C:/Users/johno/Documents/University/ETH HS22/Data Mining I/HW/HW02/data/MUTAG.mat')
label = np.reshape(mat['lmutag'], (len(mat['lmutag'], )))
data = np.reshape(mat['MUTAG']['am'], (len(label), ))


# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(
    description="Compute distance functions on time-series"
)
parser.add_argument(
    "--datadir",
    required=True,
    help="Path to input directory containing file MUTAG.mat"
)
parser.add_argument(
    "--outdir",
    required=True,
    help="Path to directory where graphs_output.txt will be created"
)
args = parser.parse_args()

# Set the paths
data_dir = args.datadir
out_dir = args.outdir

os.makedirs(args.outdir, exist_ok=True)

# Create the output file
file_name = "{}/graphs_output.txt".format(args.outdir)
f_out = open(file_name, 'w')

# names to insert into the output
lst_group = ['mutagenic', 'non-mutagenic']

# Write header for output file
f_out.write('{}\t{}\n'.format(
    'Pair of classes', 'SP'))

# Graphs with a mutagenic effect are labelled 1, all others are labelled -1
# Get position of mut and non-mut matrices
# There are 188 graphs in data
mutID = np.where(label == 1) # 0:124
non_mutID = np.where(label == -1) # 125:187

# subset to mut an non-mut matrices
mut_mat = np.take(data, mutID[0])
non_mut_mat = np.take(data, non_mutID[0])

# mutagenic:mutagenic
x = 0 # counter for getting average after summing up all K's
sumK = 0 
for mat1 in range(len(mut_mat)):
    for mat2 in range(len(mut_mat)):        
        if np.where(mut_mat == mut_mat[mat1]) == np.where(mut_mat == mut_mat[mat2]): # if they are the exact same mats don't use
            continue
        else:
            sumK += sp_kernel(floyd_warshall(mut_mat[mat1]), floyd_warshall(mut_mat[mat2])) # calculate sp_kernel            
            x += 1            
sp_ave = sumK/x
# Save the output
f_out.write('{}:{}\t{}\n'.format(lst_group[0],lst_group[0],sp_ave))

# mutagenic:non-mutagenic
x = 0
sumK = 0
for mat1 in range(len(mut_mat)):
    for mat2 in range(len(non_mut_mat)):        
        if np.where(mut_mat == mut_mat[mat1]) == np.where(non_mut_mat == non_mut_mat[mat2]):
            continue
        else:
            sumK += sp_kernel(floyd_warshall(mut_mat[mat1]), floyd_warshall(non_mut_mat[mat2]))            
            x += 1            
sp_ave = sumK/x
# Save the output
f_out.write('{}:{}\t{}\n'.format(lst_group[0],lst_group[1],sp_ave))

# non-mutagenic:non-mutagenic
x = 0
sumK = 0
for mat1 in range(len(non_mut_mat)):
    for mat2 in range(len(non_mut_mat)):        
        if np.where(non_mut_mat == non_mut_mat[mat1]) == np.where(non_mut_mat == non_mut_mat[mat2]):
            continue
        else:
            sumK += sp_kernel(floyd_warshall(non_mut_mat[mat1]), floyd_warshall(non_mut_mat[mat2]))            
            x += 1            
sp_ave = sumK/x
# Save the output
f_out.write('{}:{}\t{}\n'.format(lst_group[1],lst_group[1],sp_ave))


f_out.close()





