# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:53:46 2020

@author: Tzachy
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def covar(x):
    co = 1/x.shape[0] * (x.T@x)
    return co

if __name__ == "__main__":

    iris = pd.read_csv(r"iris.csv")
    x = iris.iloc[:, 1:5] 
    y = iris.iloc[:,5]
    y = [0 if x == 'Iris-setosa' else 1 if x == 'Iris-virginica' else 2 for x in y]
    plt.scatter(x.iloc[:,0],x.iloc[:,1], c=y)
    plt.show()
    x = x.apply(lambda x: (x-x.mean())/x.std())
    covar = covar(x)
    eigen_val, eigen_vec = np.linalg.eig(covar)
    best_val_ind = np.argsort(eigen_val, axis=0)[::-1]
    sorted_vec = eigen_vec[:,best_val_ind[:2]]
    new_df = x@sorted_vec
    print(new_df)
    
    plt.scatter(new_df.iloc[:,0],new_df.iloc[:,1], c=y)
    plt.show()

