# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 18:14:31 2022

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
print(len(data))
mutagenic_idx = np.where(label == 1)
non_mutagenic_idx = np.where(label == -1)
'''
print(mutagenic_idx)
print(non_mutagenic_idx)
'''
mutagenic_matrices = np.take(data, mutagenic_idx[0])
non_mutagenic_matrices = np.take(data, non_mutagenic_idx[0])

'''
count = 0
K_sum = 0

for matrix1 in range(len(mutagenic_matrices)):
    for matrix2 in range(len(mutagenic_matrices)):        
        if np.where(mutagenic_matrices == mutagenic_matrices[matrix1]) == np.where(mutagenic_matrices == mutagenic_matrices[matrix2]):
            continue
        else:
            K_sum += sp_kernel(floyd_warshall(mutagenic_matrices[matrix1]), floyd_warshall(mutagenic_matrices[matrix2]))
            
            count += 1
            
SP_value = K_sum/count
print(SP_value)
'''