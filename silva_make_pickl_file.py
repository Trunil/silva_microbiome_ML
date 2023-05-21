# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 17:44:19 2022

@author: trunil
"""

#save variables
import joblib

import pandas as pd

#%%
# Read the data

file = open('data\log.txt', 'w')
file.write('Started\n')
file.close()


print('\nReading otu data')
otu_table = pd.read_excel('data\silva_data.xlsx',
                          sheet_name='silva',
                          index_col='otu')
'''
otu_table is already in fractional abundance form, i.e., the sum of values 
for all taxa for each sample is 1.0
'''
print('\nReading taxa data')
taxa_data = pd.read_excel('data\silva_data.xlsx',
                          sheet_name='taxa_silva',
                          index_col='otu')

print('\nReading sample data')
sample_data = pd.read_excel('data\silva_data.xlsx',
                            sheet_name='silva_samples',
                            index_col='sample')

print('\nWriting pckl file')
joblib.dump([otu_table, taxa_data, sample_data], 'data\silva_data.pckl')

print('\nReading pckl data for verification')
silva_data = joblib.load('data\silva_data.pckl')
otu_table = silva_data[0].copy()
taxa_data = silva_data[1].copy()
sample_data = silva_data[2].copy()

del silva_data
