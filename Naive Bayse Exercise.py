# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:44:56 2020

@author: Tzachy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def pred(m_s, v):
    return 1 / (np.sqrt(2*np.pi*m_s[:,1]**2)) * np.exp(-((v-m_s[:,0])**2) / (2*m_s[:,1]**2))

def pred_prob(mean_std1, mean_std0, y, x):
    post_1 = np.sum(y)/len(y)
    post_0 = 1 - post_1
    post1 = np.log(post_1)+np.sum(np.log(pred(mean_std1, x)), axis=1)

    post0 = np.log(post_0)+np.sum(np.log(pred(mean_std0, x)), axis=1)
    print('post1', post1)
    post = []
    for i in range(len(post1)):
        if post1.iloc[i] > post0.iloc[i]:
            post.append(1)
        else:
            post.append(0)

    return post

    
if __name__ == "__main__":
    df = pd.read_csv(r"diabetes.csv")
    df.isnull().values.any()
    x_train, x_test, y_train, y_test = train_test_split(df, df['Outcome'], test_size=0.33, random_state=42)

    mean1 = x_train.loc[x_train['Outcome'] == 1].mean().drop('Outcome')
    std1 = x_train.loc[x_train['Outcome'] == 1].std().drop('Outcome')
    mean0 = x_train.loc[x_train['Outcome'] == 0].mean().drop('Outcome')
    std0 = x_train.loc[x_train['Outcome'] == 0].std().drop('Outcome')
    x_train = x_train.drop('Outcome', axis=1)
    x_test = x_test.drop('Outcome', axis=1)
    mean_std1 = np.c_[mean1, std1]
    mean_std0 = np.c_[mean0, std0]
    post = pred_prob(mean_std1, mean_std0, y_train, x_train)
    print("accuracy train", (y_train == post).mean())
    post_test = pred_prob(mean_std1, mean_std0, y_test, x_test)
    print("accuracy test", (y_test == post_test).mean())

    
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(score)

