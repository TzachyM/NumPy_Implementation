# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:46:58 2020

@author: Tzachy
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def logit(z):
    return 1 / (1+np.exp(-z))


def negL(y, g):
    return (1/len(y)) * np.sum(-y.T@np.log(g)-(1-y).T@np.log(1-g))


def grad_fun(y, h, X):
    return 1 / y.shape[0] * (h - y).T@X


def train_test(X,y):
    prec = 0.8    
    train_x = X[:np.floor(prec * X.shape[0]).astype(int)]
    test_x = X[np.floor((1-prec) * X.shape[0]).astype(int):]
    train_y = y[:np.floor(prec * y.shape[0]).astype(int)]
    test_y = y[np.floor((1-prec) * y.shape[0]).astype(int):]
    return train_x, train_y, test_x, test_y


def grad_desc (x, y, start, rate, iter_n):
    t = start.copy()
    for it in range(iter_n):
        h_x = logit(x@t)
        loss = negL(y, h_x)
        grad = grad_fun(y, h_x, x)
        t = t - rate*grad
        print(f'iteration: {it} loss: {loss} thetas: {t}')
    return t


def Accuracy(theta, test_x, test_y):
    length = test_x.shape[0]
    prediction = (theta @ test_x.T > 0.5)
    correct = prediction == test_y
    acc = np.sum(correct) / length
    return acc
    
    
if __name__ == "__main__":

    iris = load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                         columns= iris['feature_names'] + ['target'])   
    X = df.iloc[:, 0:2]
    df['target'].replace([0,1,2],[1,0,0],inplace=True)
    y = np.array(df.target)
    plt.scatter(X.iloc[:, 0:1], X.iloc[:, 1:2], c=y)
    one = np.ones((len(y), 1))

    start = np.random.randint(5, size=3)
    x = np.c_[one, X]
    train_x, train_y, test_x, test_y = train_test(x, y)
    iter_n = 2000
    rate = 0.01
    t = grad_desc (train_x, train_y, start, rate, iter_n)
    acc = Accuracy(t, test_x, test_y)
    print("Accuracy:",acc)