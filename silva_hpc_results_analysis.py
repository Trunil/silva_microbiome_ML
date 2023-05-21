# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 16:08:19 2022

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

# encoder
from sklearn.preprocessing import LabelEncoder

#models
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#train-test split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

#metrics
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

# GridSearch
from sklearn.model_selection import GridSearchCV

#save variables
import joblib


#%% data process
from silva_data_process import *

#%%

# load pickl file data

grid_search_rf = joblib.load('silva_hpc_results/silva_grid_search_rfc.pckl')

grid_search_gb = joblib.load('silva_hpc_results/silva_grid_search_gb.pckl')

shap_rfc = joblib.load('silva_hpc_results/shap_values_rfc.pckl')

data_for_shap = filtered_data


#%%
# Random forest
# initialize train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


scaler = QuantileTransformer(n_quantiles=100)
best_model_rfc = grid_search_rf.best_estimator_


#fit
best_model_rfc.fit(X_train, y_train)

#predict
y_pred = best_model_rfc.predict(X_test)

# f1 score
f1_rfc = f1_score(y_test, y_pred, average='micro')

print(f1_rfc)

#%%









