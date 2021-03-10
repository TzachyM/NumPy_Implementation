# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:04:10 2020

@author: Tzachy
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
        
    def distance_value(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
        
    def y_prediction(self, x_test):
        return np.array([self.pred(x) for x in x_test])  

    def pred(self, x):
        dist = [self.distance_value(x, x_t) for x_t in self.x_train]
        k_ind = np.argsort(dist)[:self.k]
        knn = [self.y_train[k] for k in k_ind]
        vals, freq = np.unique(knn, return_counts=True)
        most_common = vals[np.argmax(freq)]
        return most_common
    
    def accuracy(self, y, x_test):
        y_pred = self.y_prediction(x_test)
        acc = 1 - np.sum(np.abs(y-y_pred))/len(y)
        return acc
    
    
if __name__ == '__main__':
    iris = datasets.load_iris()
    x = iris.data[:, :4] 
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    k = 4
    model = KNN(k)   
    model.fit(x_train, y_train)
    print(y_test, model.y_prediction(x_test))
    print('The accuracy is:', model.accuracy(y_test, x_test))
    
    modelknn = KNeighborsClassifier(n_neighbors=3)
    modelknn.fit(x_train, y_train)
    print('Sklearn:\n', modelknn.predict(x_test))
    print(modelknn.score(x_test, y_test))