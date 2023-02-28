"""Homework 1: Distance functions on vectors.

Homework 1: Distance functions on vectors
Course    : Data Mining (636-0018-00L)

Auxiliary functions.

This file implements the distance functions that are invoked from the main
program.
"""
# Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
# Author: Bastian Rieck <bastian.rieck@bsse.ethz.ch>

import numpy
import math

def manhattan_dist(v1, v2):
    x=sum(abs(v1-v2) for v1, v2 in zip(v1,v2))
    x=float(x)
    return x  
               
def hamming_dist(v1, v2):
    x=sum(abs(v1-v2) for v1, v2 in zip(v1,v2))
    x=float(x)
    return x

def euclidean_dist(v1, v2):
    x=(sum(v1-v2 for v1, v2 in zip(v1,v2)))**0.5
    x=float(x)
    return x

def chebyshev_dist(v1, v2):
    x=max(abs(v1-v2))
    x=float(x)
    return x

def minkowski_dist(v1, v2, d):
    x=sum((abs(v1-v2))**d)**(1/d)
    x=float(x)
    return x

#TESTING
v1 = np.array([5, 9, 2, 5, 1, 6, 3, 5, 8])
v2 = np.array([1, 7, 8, 3, 4, 3, 5, 1, 9])

manhattan_dist(v1, v2)
hamming_dist(v1, v2)
euclidean_dist(v1, v2)
chebyshev_dist(v1, v2)
d=1
print(type(minkowski_dist(v1, v2, d)))
