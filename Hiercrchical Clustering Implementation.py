# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 11:47:30 2020

@author: Tzachy
"""

import numpy as np
from sklearn.datasets import load_iris


def distance(row1, row2):
    return np.sqrt(np.sum(np.square(row1-row2)))

def cluster_mat(x):
    dist = []
    for i in x:
        for j in x:
            dist.append(distance(i,j))
    dist = np.asarray(dist)
    dist = dist.reshape((len(x),len(x)))
    dist = np.where(dist == 0, np.inf, dist)
    return dist

def max_cluster(x, epoch=100):
    for i in range(epoch):
        dist = cluster_mat(x) 
        min_c = np.unravel_index(dist.argmin(), dist.shape)
        x[min_c[0]] = (x[min_c[0]]+x[min_c[1]])/2     
        x = np.delete(x, min_c[1], 0)        
    return x


if __name__ == '__main__':
    iris = load_iris()
    train = iris.data
    x = train[:, :2]
    max_clus = max_cluster(x)
    print(max_clus)
