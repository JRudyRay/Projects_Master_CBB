# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:02:32 2022

@author: johno
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import math


dtrain = pd.read_csv('C:/Users/johno/Documents/University/ETH HS22/Data Mining I/HW/HW03/data/part2/train/tumor_info.txt', sep = '\t',header = None)

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

dtrain.to_numpy()
print(nrows)
print(dtrain.shape)


value = 1
# for value in range(1,11)

for value in range(1,11): # iterate through values
    for j in range(ncols): # iterate through columns
        count = 0 # initialize counter
        for i in range(nrows): # iterate through rows
        
            if beg[i,j] == value:
                count +=1
                
        p_beg[j] = count/nrows
    print(p_beg)
        

'''
c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = 0,0,0,0,0,0,0,0,0,0

for i in range(rows_beg):
    if beg[i,0] == 1:
        c1 += 1
    if beg[i,0] == 2:
        c2 += 1
    if beg[i,0] == 3:
        c3 += 1
    if beg[i,0] == 4:
        c4 += 1
    if beg[i,0] == 5:
        c5 += 1
    if beg[i,0] == 6:
        c6 += 1
    if beg[i,0] == 7:
        c7 += 1
    if beg[i,0] == 8:
        c8 += 1
    if beg[i,0] == 9:
        c9 += 1
    if beg[i,0] == 10:
        c10 += 1

np.round(c1/rows_beg)

if beg[i,0] == 10:
    c1 += 1


print(np.round(c1/rows_beg,3))
'''

'''
dtrain = dtrain.to_numpy()
print(dtrain)
'''


#dtrain = pd.read_table('C:/Users/johno/Documents/University/ETH HS22/Data Mining I/HW/HW03/data/part2/train/tumor_info.txt')

#dtrain = np.loadtxt('C:/Users/johno/Documents/University/ETH HS22/Data Mining I/HW/HW03/data/part2/train/tumor_info.txt')
#dtrain = np.loadtxt('tumor_info.txt', delimiter= '\t', dtype=int)

#print(dtrain)
#dtrain = dtrain.to_numpy()
#print(dtrain)

# subset data into benign and malignant tumors by filtering by value in last row


#beg = dtrain[dtrain.tail(1) == 2]
#mal = dtrain[dtrain['2.1'] == 4]
#beg = dtrain[dtrain[0:,-1] == 2]
#print(beg[0,:])




#beg = dtrain[dtrain['2.1'] == 2]
#mal = dtrain[dtrain['2.1'] == 4]
'''
# intialize probability vectors
p_beg = [0] * len(beg.axes[1])
p_mal = [0] * len(mal.axes[1])

rows_beg = len(beg.axes[0])
rows_mal = len(mal.axes[0])

beg = beg.to_numpy()
mal = mal.to_numpy()


c1 = 0
for i in range(rows_beg):
    if beg[i,0] == 1:
        c1 += 1
        
print(c1/rows_beg)        

'''

#print(beg[0,:])
'''
# initialize counters
rows_beg = len(beg.axes[0])
c1=0
'''
#print(beg[beg['5'] == 1])







'''
for i in range(rows_beg):
    if beg[beg[i,'5'] == 1]:
        c1 += 1
'''

    










