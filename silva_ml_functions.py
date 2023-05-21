# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:43:37 2022

@author: trunil
"""


import numpy as np
np.random.seed(100)
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, normalize 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,plot_confusion_matrix,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy import ndimage
from sklearn import  metrics
from sklearn.metrics import roc_curve,roc_auc_score,plot_roc_curve
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans





#%% 
''' Dimension reduction functions'''

def fourier(df1, df2):
    
    train_X = np.abs(np.fft.fft(df1, axis=1))
    test_X = np.abs(np.fft.fft(df2, axis=1))
    
    return train_X, test_X

from scipy.signal import savgol_filter

def savgol(df1, df2):
    '''
    savgol filter from SciPy'''
    
    
    x = savgol_filter(df1, 21, 4, deriv=0)
    y = savgol_filter(df2, 21, 4, deriv=0)
    
    return x,y

from sklearn.preprocessing import normalize

def norm(df1, df2):
    '''
    Normalize'''
    
    
    train_X = normalize(df1)
    test_X = normalize(df2)
    
    return train_X, test_X


def robust(df1, df2):
    '''
    Return robustScaler transformed data.
    
    '''
    from sklearn.preprocessing import RobustScaler
    
    scaler = RobustScaler()
    train_X = scaler.fit_transform(df1)
    test_X = scaler.fit_transform(df2)
    
    return train_X, test_X


from imblearn.over_sampling import SMOTE

def smote(a, b):
    '''
    SMOTE sampling
    '''
    model = SMOTE()
    X, y = model.fit_resample(a, b)
    return X, y


def tsne(a, b):
    from sklearn.manifold import TSNE
    transformer = TSNE(n_components=2, metric='jaccard')
    c = transformer.fit_transform(a)
    d = transformer.fit_transform(b)
    return c,d

#%% Models

def decisionTree(train_X,train_y,test_X,test_y):
    clf = DecisionTreeClassifier()
    clf = clf.fit(train_X, train_y)
    y_pred_clf = clf.predict(test_X)
    print('*'*60)
    print("-------------------------------------------")
    print("DecisionTree Classifier")
    print(classification_report(test_y,y_pred_clf))
    print('---------------------------------------------')
    
    # plot
    print("\n\n")
    
    fig = plt.figure(figsize=(22,7))
    
    ax = fig.add_subplot(1,3,1)
    plot_confusion_matrix(clf,test_X,test_y,ax=ax)
    
    ax = fig.add_subplot(1,3,2)
    plot_roc_curve(clf,test_X, test_y,ax=ax)
    plt.plot([0, 1], [0, 1], 'k--')
    
    ax = fig.add_subplot(1,3,3)
    plot_precision_recall_curve(clf, test_X, test_y,ax=ax)
    f1=f1_score(test_y, y_pred_clf,pos_label=2)
    print("F1 score of minority class:",f1)
        
    plt.show()
    return f1

#%%

from sklearn.naive_bayes import GaussianNB

def naiveBayes(train_X,train_y,test_X,test_y):
    gnb = GaussianNB()
    gnb.fit(train_X, train_y)
    y_pred=gnb.predict(test_X)
    print('*'*60)
    print("-------------------------------------------")
    print("Gaussian NaiveBayes Classifier")
    print("")
    print(classification_report(test_y,y_pred))
    print('---------------------------------------------')
    
    # plot
    print("\n\n")
    fig = plt.figure(figsize=(22,7))
    
    ax = fig.add_subplot(1,3,1)
    plot_confusion_matrix(gnb,test_X,test_y,ax=ax)
    
    ax = fig.add_subplot(1,3,2)
    metrics.plot_roc_curve(gnb,test_X, test_y,ax=ax)
    
    plt.plot([0, 1], [0, 1], 'k--')
    
    ax = fig.add_subplot(1,3,3)
    metrics.plot_precision_recall_curve(gnb, test_X, test_y,ax=ax)
    
    f1 = metrics.f1_score(test_y, y_pred,pos_label=2)
    print("F1 score of minority class:",f1)
    
    plt.show()
    return f1

#%%
from sklearn.neighbors import KNeighborsClassifier

def knn(train_X,train_y,test_X,test_y):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_X, train_y)
    y_pred_neigh = neigh.predict(test_X)
    print('*'*60)
    print("-------------------------------------------")
    print("k-Nearest Neighbour Classifier")
    print("")
    print(classification_report(test_y,y_pred_neigh))
    print('---------------------------------------------')
    
    # plot
    print("\n\n")
    fig = plt.figure(figsize=(22,7))
    
    ax = fig.add_subplot(1,3,1)
    plot_confusion_matrix(neigh,test_X,test_y,ax=ax)
    
    ax = fig.add_subplot(1,3,2)
    metrics.plot_roc_curve(neigh,test_X, test_y,ax=ax)
    
    plt.plot([0, 1], [0, 1], 'k--')
    
    ax = fig.add_subplot(1,3,3)
    metrics.plot_precision_recall_curve(neigh, test_X, test_y,ax=ax)
    
    plt.show()
    
    f1 = f1_score(test_y, y_pred_neigh,pos_label=2)
    print("F1 score of minority class:",f1)
    
    return f1

from sklearn.ensemble import RandomForestRegressor

def randomForest(train_X,train_y,test_X,test_y):
    rnd = RandomForestClassifier()
    rnd.fit(train_X, train_y)
    y_pred_rnd = rnd.predict(test_X)
    print('*'*60)
    print("-------------------------------------------")
    print("Random Forest Classifier")
    print("")
    print(classification_report(test_y,y_pred_rnd))
    print('---------------------------------------------')
    
    # plot
    print("\n\n")
    fig = plt.figure(figsize=(22,7))
    
    ax = fig.add_subplot(1,3,1)
    plot_confusion_matrix(rnd,test_X,test_y,ax=ax)
    
    ax = fig.add_subplot(1,3,2)
    metrics.plot_roc_curve(rnd,test_X,test_y,ax=ax)
    
    plt.plot([0, 1], [0, 1], 'k--') 
    
    ax = fig.add_subplot(1,3,3)
    metrics.plot_precision_recall_curve(rnd, test_X, test_y,ax=ax)
    
    plt.show()
    
    f1 = f1_score(test_y, y_pred_rnd,pos_label=2)
    print("F1 score of minority class:",f1)
    
    return f1


