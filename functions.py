# -*- coding: utf-8 -*-
"""
Useful functions for crceller work

Created on Tue Aug  2 16:19:58 2022

@author: trunil
"""

#%% Import functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#pipeline
from sklearn.pipeline import Pipeline

#scalers
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

#models
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#train-test split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

#metrics
from sklearn.metrics import accuracy_score

# GridSearch
from sklearn.model_selection import GridSearchCV

#%% functions

def reset_train_test_data(X,y, stratified = False):
    
    if stratified==True:
        X = X.reset_index()
        X = X.drop(columns='index')

        split = StratifiedShuffleSplit(n_splits=1,
                                       test_size=0.2,
                                       random_state=24)
        
        for train_index, test_index in split.split(X,y):
                        
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
       
            
    else:
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
        
    return X_train, X_test, y_train, y_test
   


# In[18]:


def run_pipeline(X_train, X_test, y_train, y_test, pipe):
    

    # The pipeline can be used as any other estimator
    # and avoids leaking the test set into the train set

    pipe.fit(X_train, y_train)

    pred_train = pipe.predict(X_train)
    accuracy_train = accuracy_score(pred_train,y_train)

    pred_test = pipe.predict(X_test)
    accuracy_test = accuracy_score(pred_test,y_test)
    
    print('\n\n')

    print(f'Accuracy on training data is : {accuracy_train} \nAccuracy on test data is: {accuracy_test}')
    
    return pipe

#%% Funtion for getting genus names from id

def get_genus_names(result_df, data_with_genus):
    '''
    result_df = the dataframe from which you want to extract names of genus
    data_with_genus = this has ids and corresponding genus
    '''

    ids = result_df.index.to_numpy()
    genus = data_with_genus['genus'][ids]
    
    return genus


def check_genus_names_1(ids, data_with_genus):
    ids_data = data_with_genus.index
    for i in ids:
        if i in ids_data:
            print(i, 'True')
        else:
            print(i, 'False********')

#%% get otu names
def get_otu_names(result_df, data_with_otus):
    '''
    result_df = the dataframe from which you want to extract names of genus
    data_with_genus = this has ids and corresponding genus
    '''

    ids = result_df.index.to_numpy()
    otus = data_with_otus['taxon_name'][ids]
    
    return otus