# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:39:57 2021

@author: Tzachy
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


class MyDBSCAN:
    
    def __init__(self, x, epsilon=0.5, neighbours_num=3):
        self.epsilon = epsilon
        self.x = x
        self.k = neighbours_num
        self.visited = np.empty(150, dtype=object)
        self.clusters = np.empty(150, dtype=object)
 
    def regionQuery(self, point_index):
        neighbours = []
        for i in range(len(self.x)):
            if np.linalg.norm(self.x[point_index]-self.x[i]) <= self.epsilon:
                neighbours.append(i)
        return neighbours 
            
    def expandCluster(self, core, cluster, neighbours):
        self.clusters[core] = cluster
        n = 0
        stop_search = len(neighbours)
        while n != stop_search:
            if neighbours[n] not in self.visited:      
                self.visited[neighbours[n]] = neighbours[n]
                sphere_points = self.regionQuery(neighbours[n])
                if len(sphere_points) >= self.k:
                    neighbours += sphere_points
                    stop_search = len(neighbours)
                if self.clusters[neighbours[n]] == None:
                    self.clusters[neighbours[n]] = cluster
            n += 1
                
    def expend(self):
        cluster = 0 
        for i in range(len(x)):
            if i not in self.visited:
                self.visited[i] = i
                neighbours = self.regionQuery(i)
                if len(neighbours) < self.k:
                    continue
                else:   
                    self.expandCluster(i, cluster, neighbours)
                    cluster += 1
        self.clusters = np.array([-1 if x==None else x for x in self.clusters])
                    
        
if __name__ == "__main__":
    data = load_iris()
    x = data.data
    y = data.target
    normal = MinMaxScaler()
    x = normal.fit_transform(x)
    epsilon = 0.3   
    neighbours_num = 3
    clusters = MyDBSCAN(x, epsilon, neighbours_num)
    clusters.expend()
    print("My DBSCAN:\n",clusters.clusters)
    sk_clusters = DBSCAN(eps=epsilon, min_samples=neighbours_num)
    print(20*'*',sk_clusters.fit(x),20*'*')
    print(sk_clusters.fit_predict(x))
    sklearn = sk_clusters.fit_predict(x)
    plt.scatter(x=x[:,0], y=x[:,1], c=clusters.clusters)
    plt.figure()
    plt.scatter(x=x[:,0], y=x[:,1], c=sklearn)

