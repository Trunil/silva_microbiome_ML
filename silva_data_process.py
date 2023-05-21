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


#%%
# Read the data

file = open('output\log.txt', 'w')
file.write('Started\n')
file.close()

# read data from pckl file
silva_data = joblib.load('data\silva_data.pckl')
otu_table = silva_data[0].copy()
taxa_data = silva_data[1].copy()
sample_data = silva_data[2].copy()

del silva_data

#%%

# group otu_table by genus
otu_table['genus'] = taxa_data['Genus']
otu_table_genus_grouped = (otu_table
                           .groupby('genus')
                           .sum()
                           .reset_index())


# keep those genus that are present in 5% of samples
num_samples = otu_table_genus_grouped.shape[1]-1
            # not counted 'genus' column

# should be present in 50% of samples
positive_data_points_bool = (otu_table_genus_grouped
                .iloc[:,1:]
                > 0)

num_post_value_in_row = positive_data_points_bool.sum(axis=1)
ind = np.where(num_post_value_in_row >= int(0.5*num_samples))
filtered_data = (otu_table_genus_grouped
                 .iloc[ind]
                 .set_index('genus')
                 .T
                 .rename_axis('sample', axis=1)
                 )


# make labels
enc = LabelEncoder()
sample_data['Group_encoded'] = enc.fit_transform(sample_data['Group'])

#%% 
'''Select only 'Healthy' and 'Stage_III_IV'

'''

cat_dict = {1: 'Healthy',
	3: 'Stage_0',
	4: 'Stage_III_IV',
	5: 'Stage_I_II'}


filtered_data['Group'] = sample_data['Group_encoded']


new_filt_data = filtered_data.loc[(filtered_data['Group']==enc.transform(['Healthy'])[0]) |
                                   (filtered_data['Group']==enc.transform(['Stage_III_IV'])[0])]

#for 1,3,4,5
# new_filt_data = filtered_data.loc[(filtered_data['Group'] == enc.transform(['Healthy'])[0]) |
#                                   (filtered_data['Group'] == enc.transform(['Stage_0'])[0]) |
#                                   (filtered_data['Group'] == enc.transform(['Stage_III_IV'])[0]) |
#                                   (filtered_data['Group'] == enc.transform(['Stage_I_II'])[0])]

#new_filt_data = filtered_data.copy()

cols = new_filt_data.columns[0:-1]

#%% Make features

filtered_data = new_filt_data.drop('Group', axis=1)

X = (new_filt_data
     .drop('Group', axis=1)
     .to_numpy()
     )

y = new_filt_data['Group'].to_numpy()

file = open('output\log.txt', 'a')
file.write('\n\nFeatures and labels created...\n')
file.close()

