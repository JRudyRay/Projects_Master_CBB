"""Skeleton file for your solution to the shortest-path kernel."""
import numpy as np

def floyd_warshall(A):
    n = A.shape[0]
    A = A.astype('float64') # must be float so I can set values = np.inf
    for i in range(n):
        for j in range(n):
            if A[i,j] == 0 and i!=j:
                A[i,j] = np.inf
    # pseudocode from slides            
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if A[i,j] > A[i,k] + A[k,j]:
                    A[i,j] = A[i,k] + A[k,j]
    return A


def sp_kernel(S1, S2):
    sumK = 0
    # turn upper triangle of matrix into a list, since matrix is symmetric
    S1_ind = np.triu_indices_from(S1)
    upper1 = S1[S1_ind]
    n = len(upper1)
    
    S2_ind = np.triu_indices_from(S2)
    upper2 = S2[S2_ind]
    m = len(upper2)
    # compare each value in S1 to each value in S2
    for i in range(n):
        for j in range(m):
            if upper1[i] == upper2[j]:
                sumK += 1
    sumK = float(sumK)
    return sumK
