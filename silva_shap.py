# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 17:07:30 2022

@author: trunil
"""

import numpy as np
np.random.seed(42)
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

#pickle for importing saved model
import pickle
import joblib
# for loading model

# SHAP
import shap



#%%
file = open('log_shap.txt', 'w')
file.write('Data process started\n')
file.close()

# Run the data processing and model training files and import functions
from silva_data_process import *

from functions import *

#from silva_ml import *

file = open('log_shap.txt', 'a')
file.write('Data process finished\n')
file.close()
#%%
file = open('log_shap.txt', 'a')
file.write('loading grid search result.\n')
file.close()

grid_search = joblib.load('silva_grid_search_rfc.pckl')

#%%
#training and test dataset reset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%% Pipeline
#scaler = QuantileTransformer(n_quantiles=100)

pipe = grid_search.best_estimator_

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# acc = accuracy_score(y_test, y_pred)
# print(acc)
#%% shapley values
#shap.initjs()

#%% Shapley values for multiple samples
file = open('log_shap.txt', 'a')
file.write('Calculating shap values.\n')
file.close()

data_for_shap = filtered_data.iloc[0:10,:]

explainer = shap.KernelExplainer(model = pipe.predict, data = X,
                                 link = "identity")

#% select samples for which you want to calculate shapley values
#sel_X_for_shapley = X_test.iloc[0:10,:] 
# for selected samples use X_test.iloc[0:10,:]

#calculate shapley value
shap_values = explainer.shap_values(X = data_for_shap, nsamples = 300)

joblib.dump(shap_values, 'shap_values_rfc.pckl')
joblib.dump(data_for_shap, 'data_for_shap.pckl')

file = open('log_shap.txt', 'a')
file.write('Calculated shap values.\n')
file.close()

# In[259]:


# print the JS visualization code to the notebook
#shap.initjs()

#print(f'Current Label Shown: {list_of_labels[current_label.value]}\n')


# shap.summary_plot(shap_values = shap_values,
#                   features = data_for_shap,
#                   max_display=10
#                   )


# fig , ax = plt.subplots(1,1)
# ax=p
# plt.show()



