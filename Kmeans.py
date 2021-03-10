# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 11:13:44 2020

@author: Tzachy
"""

import numpy as np
from sklearn.datasets import load_iris


def distance(row1, row2):
    return np.sqrt(np.sum(np.square(row1-row2)))

def find_cluster(x, k):
    cluster_list = []
    for row in x:
        dist = []
        for kmean in k:        
            dist.append(distance(row, kmean))
        min_dist_index = dist.index(min(dist)) 
        cluster_list.append(min_dist_index)
    return np.array(cluster_list)

def calc_cluster(x, cluster_num=3, alpha=0.001):
    index = np.random.choice(x.shape[0], cluster_num, replace=False)  
    k = x[index,:] 
    temp_k = np.zeros((cluster_num, x.shape[1]))
    while np.sum(k-temp_k) > alpha:
        temp_k = k    
        cluster_idx = find_cluster(x, k)
        for i in range(cluster_num):
            k[i] = (x[cluster_idx==i]).mean()
    return cluster_idx
            
            
if __name__ == '__main__':
    iris = load_iris()
    train = iris.data
    cluster_idx = calc_cluster(train)
    print(cluster_idx)