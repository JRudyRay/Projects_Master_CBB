# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 23:00:37 2022

@author: johno
"""

D = 60
Nmax = 1000
PN = 1/Nmax
PNDs = [0] * ((Nmax-D)+1)

ssum = 0
for N in range(D,Nmax+1):
    x = (1/N)*PN
    ssum += x


for i in range(D,Nmax+1):
    PDN = 1/i
    PND = (PN*PDN)/ssum
    PNDs[i-D] = PND



for i in PNDs:
    if i == max(PNDs):
        N = int(i + 60)
        print('The maximal posterior probability is obtained at N =', N)
        print('where the posterior probability is ', max(PNDs))
    



### Expected value

END = 0

for i in range(D,Nmax+1):
    PND = ((1/i)*(1/Nmax))/ssum
    x = i*PND
    END += x
END = int(END)
print('')    
print('The expected value of the posterior distribution is :',END)
    
