"""Homework 1: Distance functions on vectors.

Homework 1: Distance functions on vectors
Course    : Data Mining (636-0018-00L)

Auxiliary functions.

This file implements the distance functions that are invoked from the main
program.
"""
# Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
# Author: Bastian Rieck <bastian.rieck@bsse.ethz.ch>

import numpy as np
import math

def manhattan_dist(v1, v2):
    return sum(abs(v1-v2))

def hamming_dist(v1, v2):
    new_vec = [0]*len(v1)
    for i in range(len(v1)):
        if v1[i] > 0:
            x = 1
        else:
            x = 0
        if v2[i] > 0:
            y = 1
        else:
            y = 0
        new_vec[i] = abs(x-y)
    return float(sum(new_vec))

def euclidean_dist(v1, v2):
    return (sum((v1-v2)**2))**0.5

def chebyshev_dist(v1, v2):
    return max(abs(v1-v2))  

def minkowski_dist(v1, v2, d):
    return sum((abs(v1-v2))**d)**(1/d)  


""" Discarded part of Hamming_distance due to strange abnormalities, i'ts like it rendered the original vector or something like that
    new_v1 = v1
    new_v2 = v2
    def binarise(vec):
        for i in range(0,len(vec)):
            if vec[i] > 0:
                vec[i] = 1
            else:
                vec[i] = 0
        return vec
    v1 = binarise(new_v1)
    v2 = binarise(new_v2)
"""
