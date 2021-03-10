# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:31:36 2020

@author: Tzachy
"""

import numpy as np
from sklearn.datasets import load_iris


def high_dim(x, x_std=1):
    p = np.zeros((x.shape[0], x.shape[0]))
    for i in range(len(x)):
        for j in range(len(x)):
            if j != i:
                p[i,j] = np.exp((-(np.linalg.norm(x[i]-x[j]))**2)/2*x_std**2) / np.sum(np.exp(-(np.linalg.norm(x[i]-x[j]))**2)/2*x_std**2)
    return p


def low_dim(x):
    q = np.zeros((x.shape[0], x.shape[0]))
    for i in range(len(x)):
        for j in range(len(x)):
            if j != i:
                q[i,j] = np.sum(1+(np.linalg.norm(x[i]-x[j]))**2) / (1+(np.linalg.norm(x[i]-x[j]))**2)
    return q


def loss_func(x, p, q, x_std=1):
    loss = 0
    for i in range(x.shape):
        for j in range(x.shape):
            if p[i,j] != 0 or q[i,j] != 0:
                loss += np.sum(p[i,j] *np.log(p[i,j] / q[i,j]))
    return loss


def grad_calc(x, p, q):

    for i in range(len(x)):
            grad = 4*np.sum((p-q)@(x[i]-x)/(1+np.linalg.norm(x[i]-x))) 


def training(x):
    loss = loss_func(x)
    p =  high_dim(x)
    q = low_dim(x)


if __name__ == '__main__':

    iris = load_iris()
    x = iris.data
    print(low_dim(x))
