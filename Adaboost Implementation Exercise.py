# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:09:33 2020

@author: Tzachy
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier


class AdaBoost:
    def fit(self,x,y, epoch=50):
        row_num = x.shape[0]
        self.sample_w = np.zeros((epoch,row_num))
        self.stamps = np.zeros(epoch, dtype=object)
        self.stamp_w = np.zeros(epoch)
        self.errors = np.zeros(epoch)
        self.sample_w[0] = np.ones(row_num)/row_num
        for i in range(epoch):
            current_sample_w = self.sample_w[i]
            stamp = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            stamp = stamp.fit(x, y, sample_weight=(current_sample_w))
            stamp_pred = stamp.predict(x)
            error = current_sample_w[stamp_pred != y].sum()
            stamp_w = np.log((1-error)/error)/2
            new_sample_w = current_sample_w * np.exp(-stamp_w*y*stamp_pred)
            new_sample_w /= new_sample_w.sum()
            if i+1<epoch:
                self.sample_w[i+1] = new_sample_w
            self.stamps[i] = stamp
            self.stamp_w[i] = stamp_w
            self.errors[i] = error
    
    def predict(self, x):
        y_pred = np.array([stamp.predict(x) for stamp in self.stamps])
        return np.sign(self.stamp_w @ y_pred)
    
    def score(self,x,y):
        pred = self.predict(x)
        return 1 - np.sum(np.abs(pred-y))/len(y)
    
if __name__ == '__main__':
    df = pd.read_csv(r'heart.csv')
    y = df['target']
    x = df.drop('target', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    
    model = AdaBoost()
    model.fit(x_train,y_train)
    print(model.predict(x_test))
    print("my ada", model.score(x_test,y_test))
  
    mode = AdaBoostClassifier()  
    mode.fit(x_train,y_train)
    print(mode.predict(x_test))
    print(mode.score(x_test,y_test))              
