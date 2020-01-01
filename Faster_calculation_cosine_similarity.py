# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 12:51:54 2020

@author: tamohant
"""
import numpy as np

#Building the matrix
u = np.random.rand(50)
print(u)
v = np.random.rand(50)
print(v)

#Cosine Similarity traditional way
def cosine_similarity(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta


cosine_similarity(u, v)


#Cosine similarity caclulation using numba
    
from numba import jit

@jit(nopython=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta

cosine_similarity_numba(u, v)

k = 10 # Change this value and run the below cells to experiment with different number of computations.

import time


for i in range(k):
    start = time.time()
    cosine_similarity(u, v)
    end = time.time()
    print("Time taken:", end-start)


for i in range(k):
    start = time.time()
    cosine_similarity_numba(u, v)
    end = time.time()
    print("Time taken:", end-start)