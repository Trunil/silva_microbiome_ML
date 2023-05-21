# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 23:57:27 2022

@author: trunil
"""
#%% File for writing details
file = open('silva_output.txt', 'w')
file.close()

#%%
# read data and make important variables
from silva_data_process import *

#%%

#initialize data
X_train, X_test, y_train, y_test = reset_train_test_data(X, y_data, stratified=True)

file = open('silva_output.txt', 'a')
file.write(f'Shape X_train = {X_train.shape} \n')
file.write(f'Shape X_test = {X_test.shape} \n')
file.write(f'Shape y_train = {y_train.shape} \n')
file.write(f'Shape y_test = {y_test.shape} \n')
file.close()
#%% SVC model GridSearchCV
scaler = QuantileTransformer(n_quantiles=100)
model_svc = SVC()



#setup grid
grid_params = {'model__C': [0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10, 100],
               'model__gamma': [0.0001, 0.001, 0.01, 0.1, 1],
               'model__kernel': ['rbf']}

pipe = Pipeline([('scaler', scaler), ('model', model_svc)])

grid = GridSearchCV(pipe,
                   grid_params,
                   cv=5,
                   scoring='accuracy',
                   return_train_score=False,
                   verbose=1)


svc_grid_search = grid.fit(X_train, y_train)

# write best estimator to file
file = open('silva_output.txt', 'a')
file.write(f"\n\n{'*'*20} \nBest Estimator: \n")
file.write(str(svc_grid_search.best_estimator_))
file.close()

#predict
y_pred = svc_grid_search.predict(X_test)
acc_svc = accuracy_score(y_test, y_pred)
print(f'{acc_svc=}')


# store
import joblib
joblib.dump(grid_search, 'silva_svc.pckl')
joblib.dump(pipe, 'pipeline.pckl')


#%% Random forest
from sklearn.ensemble import RandomForestClassifier

scaler = QuantileTransformer(n_quantiles=100)
rfc=RandomForestClassifier(random_state=42)

pipe = Pipeline([('scaler', scaler), ('model', rfc)])

# param_grid = { 
#     'n_estimators': [100, 200, 400],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'criterion' :['gini', 'entropy', 'log_loss']
# }

param_grid = { 
    'n_estimators': [100, 200, 400],
    
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

rfc_grid_search = CV_rfc.fit(X_train, y_train)

y_pred = CV_rfc.predict(X_test)
acc_rfc = accuracy_score(y_test, y_pred)
print(f'{acc_rfc=}')



# write accuracies
file = open('silva_output.txt', 'a')
file.write('\n\n**************************')
file.write('Acuracies\n')
file.write(f'{acc_svc=}\n')
file.write(f'{acc_rfc=}\n')

file.close()
