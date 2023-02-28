"""
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
    return float(np.sum(abs(v1 - v2)))

def hamming_dist(v1, v2):
    return float(np.sum((v1 > 0.0) != (v2 > 0.0)))

def euclidean_dist(v1, v2):
    return float(math.sqrt(np.sum((v1 - v2)**2)))

def chebyshev_dist(v1, v2):
    return float(np.max(abs(v1 - v2)))

def minkowski_dist(v1, v2, d):
    return float(np.sum(abs(v1 - v2)**d)) ** (1.0 / d)
