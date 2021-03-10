# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:09:15 2021

@author: Tzachy
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler


class MyGMM:
    
    def __init__(self, x, k, covar):
        self.covar = covar
        self.x = x
        self.centroids = x[np.random.randint(x.shape[0], size=k)]
        self.w = (np.random.dirichlet(np.ones(k),size=1)).T
        self.respons_mat = np.zeros((len(x),k))
        self.log_like = 0
        
    def point_prob(self,x,center, covar):
        center = np.array(center).reshape((-1,1))
        x = np.array(x).reshape((-1,1))
        point = np.exp(-0.5*(x-center).T @ np.linalg.inv(covar)@(x-center)) / ((2*np.pi)**(self.x.shape[1]/2) * np.sqrt(np.linalg.det(covar)))
        return point
    
    def expect_prob(self):
        for i in range(len(x)):
            for j in range(len(self.w)):
                gauss = self.point_prob(self.x[i], self.centroids[j], self.covar[j])
                self.respons_mat[i,j] = (self.w[j] * gauss)
            likelihood = np.sum(self.respons_mat[i])
            self.log_like += np.log(likelihood)
            self.respons_mat[i] = self.respons_mat[i]/likelihood
        
        self.update()

    def update(self):
        for i in range(len(self.w)):   
            q = self.respons_mat[:,i].reshape((-1,1))
            self.w[i] = np.sum(q) / np.sum(self.respons_mat) 
            self.centroids[i] = np.sum(q*self.x) / np.sum(q)
            self.covar[i] = (self.x-self.centroids[i]).T@(q*(self.x-self.centroids[i])) / np.sum(q)

    def gmm(self, epoch, log_like_stop):  
        for i in range(epoch):
            self.expect_prob()
            if log_like_stop <= self.log_like:
                break
        self.clusters()
        
    def clusters(self):  
        final_clusters = []
        for i in range(len(self.x)):
            final_clusters.append((np.argmax(self.respons_mat[i])))
        print(final_clusters)
            

if __name__ == '__main__':

    print("\U0001F602") 
    data = load_iris()
    x = data.data
    normal = MinMaxScaler()
    x = normal.fit_transform(x)
    k = 3
    epoch = 10000
    log_like = 0.0001
    covar = np.cov(x.T)
    covar = np.dstack((covar,covar,covar)).T
    model = MyGMM(x, k, covar)
    model.gmm(epoch, log_like)
    print("🧐")