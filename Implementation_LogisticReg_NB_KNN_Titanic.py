# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:04:10 2020

@author: Tzachy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class KNN:
    
    def __init__(self,k=3):
        self.k = k
    
    def fit(self,x,y):
        self.x_train = x
        self.y_train = y
    
    def distance_value(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def predict(self,x_test):
        predicted_labels = [self.k_near(x) for x in x_test]
        return np.array(predicted_labels)
    
    def k_near(self,x):
        distances = [self.distance_value(x, x_t ) for x_t in self.x_train]       
        k_ind = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[k] for k in k_ind]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    

    def accuracy(self,x_test, y):
        y_pred = self.predict(x_test)
        return 1 - np.sum(np.abs(y_pred - y)) / len(y)

class NB:

    def probality(self, mean, std, x):
        return np.exp((-(x-std)**2)/(2*(mean)**2)) / (np.sqrt(2*np.pi*std))
    
    def mean_std(self, x_train, y_train):
        mean1 = x_train.loc[y_train == 1].mean()
        std1 = x_train.loc[y_train == 1].std()
        mean0 = x_train.loc[y_train == 0].mean()
        std0 = x_train.loc[y_train == 0].std() 
        return mean1, std1, mean0, std0
    
    def probabilty_nb(self, x, y):
        mean1, std1, mean0, std0 = self.mean_std(x, y)
        prob1 = len(mean1)/(len(mean1)+len(mean0))
        prob0 = 1-prob1
        probality1 = np.log(prob1)+np.sum(np.log(self.probality(mean1, std1, x)), axis=1)
        probality0 = np.log(prob0)+np.sum(np.log(self.probality(mean0, std0, x)), axis=1)
        post = []
        for i in range(len(probality1)):
            if probality1.iloc[i] > probality0.iloc[i]:
                post.append(1)
            else:
                post.append(0)
        return post
    
    def accuarcy(self, y_pred, y_test):
        return 1 - np.sum(np.abs(y_pred - y_test)) / len(y_test)
    
    
class LogisticReg:
    
    def model(self, x, theta):
        return x @ theta
    
    def sigmoid(self, h_x):
        return 1/(1+np.exp(-h_x))
    
    #def likelihood(self, y, h_x):
     #   y_pred = self.sigmoid(h_x)
      #  return np.sum(y * log(y_pred) + (1-y) * log (1 - y_pred)), y_pred
    
    def grad_ascent(self, y, h_x, x):
        #like, h_x = self.likelihood(y, h_x)
        #print(like)
        return ((h_x - y) @ x) / len(y)
    
    def min_theta(self, y, x, alpha, epochs=2000):
        theta = np.random.rand(x.shape[1])
        for i in range(epochs):
            h_x = self.model(x, theta)
            y_pred = self.sigmoid(h_x)
            gradient = self.grad_ascent(y, y_pred, x)
            theta = theta - alpha * gradient
        self.theta = theta
        return theta
    
    def accuracy(self, x_test, y_test, threshold=0.5):
        h_x = self.model(x_test, self.theta)
        y_pred = self.sigmoid(h_x)
        prediction = np.zeros(len(y_test))
        prediction[y_pred >= threshold] = 1
        return 1-(np.sum(np.abs(prediction-y_test))/len(y_test))
    
if __name__ == '__main__':
    
    
    url = r'https://gist.githubusercontent.com/michhar/2dfd2de0d4f8727f873422c5d959fff5/raw/fa71405126017e6a37bea592440b4bee94bf7b9e/titanic.csv'   
    df = pd.read_csv(url)   
    df['Sex'] = [1 if x=='male' else 2 for x in df['Sex']]   
    df = df.drop('Cabin',axis=1)   
    df = df.dropna()   
    X = df[['Pclass', 'Age', 'Sex','Fare']]   
    y = df['Survived'] 
    scaler = StandardScaler()
    scaler.fit(X)
    scaler.transform(X)
    X = X.values   
    y = y.values   
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    
    
    
    #KNN   
    print('---------KNN----------')
    model = KNN(3)   
    model.fit(X_train,y_train)
    
    #print(model.predict(X_test))
    #print(y_test)
    print('manual KNN accuarcy is ' , model.accuracy(X_test, y_test))
    
    modelknn = KNeighborsClassifier(n_neighbors=3)
    modelknn.fit(X_train, y_train)
    #print(modelknn.predict(X_test))
    print("Sklearn KNN",modelknn.score(X_test, y_test))
    
    
    #Naive Bayes
    print('\n---------Naive Bayes----------')
    x = df[['Pclass', 'Age', 'Sex','Fare']]  
    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2)
    model_nb = NB()
    post = model_nb.probabilty_nb(x_train, y_train)
    print("manual accuracy train", (y_train == post).mean())
    post_test = model_nb.probabilty_nb(x_test, y_test)
    print("manual accuracy test", (y_test == post_test).mean())
    
    modelNB_multi = MultinomialNB()
    modelNB_multi.fit(X_train, y_train)
    #print(modelNB_multi.predict(X_test))
    print("Sklearn NB-multi:", modelNB_multi.score(X_test, y_test))
    
    modelNB_gaus = GaussianNB()
    modelNB_gaus.fit(X_train, y_train)
    #print(modelNB_gaus.predict(X_test))
    print("Sklearn NB-gauss:", modelNB_gaus.score(X_test, y_test))
    
    
    #Logistic regression
    print('\n---------Logistic regression----------')    
    model_logistic = LogisticReg()
    alpha = 0.01
    model_logistic.min_theta(y_train, x_train, alpha)
    print("manual logistics reg:",model_logistic.accuracy(x_test, y_test))
    
    model_log = LogisticRegression()
    model_log.fit(X_train, y_train)
    #print(model_log.predict(X_test))
    print("Sklearn logistic:", model_log.score(X_test, y_test))