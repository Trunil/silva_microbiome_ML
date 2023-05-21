# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 16:02:10 2022

@author: trunil
"""


import numpy as np
np.random.seed(100)
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
from sklearn.ensemble import RandomForestClassifier

# report
from sklearn.metrics import classification_report

#train-test split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

#metrics
from sklearn.metrics import accuracy_score

# GridSearch
from sklearn.model_selection import GridSearchCV

#pickle for importing saved model
import joblib


#%%

# Run the data processing and model training files and import functions
from silva_data_process import *

from silva_ml_functions import *

#%%

def run_rfc(X_train, X_test, y_train, y_test):
    scaler = QuantileTransformer(n_quantiles=100)
    model_rfc = RandomForestClassifier()
    
    pipe_rfc = Pipeline([('scaler', scaler),
                         ('model', model_rfc)])
    
    param_grid = {
        'model__max_depth': [1,5, 8, 10, 15, 20],
        'model__n_estimators': [1,5, 8, 10, 15, 20],
        'model__min_samples_split': [2, 5, 10]
        }
        
    
    #print('*'*5)
    
    CV_rfc = GridSearchCV(estimator=pipe_rfc,
                          param_grid=param_grid,
                          scoring='f1_micro',
                          cv= 10,
                          return_train_score=False,
                          verbose=1)
    
    #print('\n\n grid search starting')
    grid_search_rfc = CV_rfc.fit(X_train, y_train)
    
    # save rfc grid search
    joblib.dump(grid_search_rfc, 'silva_grid_search_rfc.pckl')
    
    #print('\n\n grid search done')
    
    
    # define model
    best_model_rfc = grid_search_rfc.best_estimator_
    
    #fit
    best_model_rfc.fit(X_train, y_train)
    
    #predict
    y_pred = best_model_rfc.predict(X_test)
    
    # f1 score
    f1_rfc = f1_score(y_test, y_pred, average='micro')
    
    print('F1 score with feature selection is: ', f1_rfc)


#%%
''' Chi-squared feaure selection'''
print('\n\nChi-squared feaure selection')

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# scale
scaler = QuantileTransformer(n_quantiles=100)
X_train = scaler.fit_transform(X_train)

num_feats = 10

#X_norm = MinMaxScaler().fit_transform(X_train)
chi_selector = SelectKBest(chi2, k=num_feats)
chi_selector.fit(X_train, y_train)
chi_support = chi_selector.get_support()
chi_feature = filtered_data.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')

#%%% ml with feature selected data

sel_data = filtered_data[chi_feature]

X = sel_data.to_numpy()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#%% transformation

# fourier
X_train, X_test = fourier(X_train, X_test)

# tsne transform
X_train, X_test = tsne(X_train, X_test)


#%%
print('\n\nChi-squared : ML-RFC')

run_rfc(X_train, X_test, y_train, y_test)
    
#%%
#%% 
'''Recursive feature elimination'''
# can be used with any model
print('\n\nRecursive feature elimination')

from sklearn.feature_selection import RFE

X = (filtered_data.copy()
      .to_numpy()
      )



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# scale
scaler = QuantileTransformer(n_quantiles=100)
X_train = scaler.fit_transform(X_train)


num_feats = 10

rfe_selector = RFE(estimator=RandomForestClassifier(),
                   n_features_to_select=num_feats,
                   step=10,
                   verbose=5)

rfe_selector.fit(X_train, y_train)
rfe_support = rfe_selector.get_support()
rfe_feature = filtered_data.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')

#%%% ml with feature selected data

sel_data = filtered_data[rfe_feature]

X = sel_data.to_numpy()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%% transformation

# fourier
X_train, X_test = fourier(X_train, X_test)

# tsne transform
X_train, X_test = tsne(X_train, X_test)


#%%
print('\n\nRFE : ML-RFC')

run_rfc(X_train, X_test, y_train, y_test)

#%%


