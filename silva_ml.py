# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 22:05:26 2022

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

from silva_ml_functions import *


#%%
#initialize training and test datasets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



res_dict = {}


file = open('data\log.txt', 'a')
file.write('train test datasets created.\n')
file.close()

#%%
# random forest
file = open('data\log.txt', 'a')
file.write('\n\nRandom forest.\n')
file.close()


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

scaler = QuantileTransformer(n_quantiles=100)
model_rfc = RandomForestClassifier()

pipe_rfc = Pipeline([('scaler', scaler), ('model', model_rfc)])

#acc_rfc = run_model(pipe_rfc, X_train, X_test, y_train, y_test)

#fit
pipe_rfc.fit(X_train, y_train)

#predict
y_pred = pipe_rfc.predict(X_test)

#accuracy
acc_rfc = accuracy_score(y_test, y_pred)

#confucion matrix
cf = confusion_matrix(y_test, y_pred)
#print(cf)


# f1 score
f1_rfc = f1_score(y_test, y_pred, average='micro')


#(y_true, y_pred, *,
#labels=None, pos_label=1,
#average='binary', sample_weight=None, zero_division='warn')[source]¶

res_dict['rfc'] = f1_rfc

    

#%%


file = open('output\log.txt', 'a')
file.write('Random forest cv started\n')
file.close()



# Random forest Gridsearch
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score

scaler = QuantileTransformer(n_quantiles=100)
model_rfc = RandomForestClassifier()

pipe_rfc = Pipeline([('scaler', scaler),
                     ('model', model_rfc)])

param_grid = {
    'model__max_depth': [1,10,40, 50, 100, 150, 200],
    'model__n_estimators': [1,10, 50, 100, 200],
    'model__min_samples_split': [2, 10, 50]
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
    
    
print(f'{f1_rfc=}')

file = open('output\log.txt', 'a')
file.write('Random forest cv completed\n')
file.close()


#%%

file = open('output\log.txt', 'a')
file.write('Gradient boost cv started\n')
file.close()

# initialize train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# gradient boost

from sklearn.ensemble import GradientBoostingClassifier

scaler = QuantileTransformer(n_quantiles=100)
model_gb = GradientBoostingClassifier()

pipe_gb = Pipeline([('scaler', scaler), ('model', model_gb)])

# fit model for getting the hyperparameter variable names
pipe_gb.fit(X_train, y_train)
#pipe_gb.get_params()

# grid parameters

param_grid = {
    'model__max_depth': [1,5,10],
    'model__n_estimators': [5]
    }



CV_gb = GridSearchCV(estimator=pipe_gb,
                      param_grid=param_grid,
                      scoring='f1_micro',
                      cv= 10,
                      return_train_score=False,
                      verbose=1)


print('\n\n grid search starting')
grid_search_gb = CV_gb.fit(X_train, y_train)

# save rfc grid search
joblib.dump(grid_search_gb, 'silva_grid_search_gb.pckl')

#print('\n\n grid search done')


# define model
best_model_gb = grid_search_gb.best_estimator_

#fit
best_model_gb.fit(X_train, y_train)

#predict
y_pred = best_model_gb.predict(X_test)

# f1 score
f1_gb = f1_score(y_test, y_pred, average='micro')
    
print(f'{f1_gb=}')
file = open('output\log.txt', 'a')
file.write('Gradient boost cv completed.\n')
file.close()


#%%

