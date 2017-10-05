# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
Chase Ginther
"""

import pandas as pd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

os.chdir("C:\\Users\\Chase\\Desktop\\School\\Fall 2017\\CMDA 4684\\County")

#read in data sources
data = pd.read_csv("data\ohio.csv")
unemployment = pd.read_csv("data\OhioUnemploymentReport.csv")
education = pd.read_csv("data\EducationReport.csv")

#column name cleanup
education['Name'] = education['Name'].apply(lambda x: x.upper())
unemployment.rename(columns={'Unnamed: 0': 'County'}, inplace=True)

#convert the hh income column from a string to an int
unemployment['Median HH Income'] = unemployment['Median Household Income (2015)']\
                                        .str.replace('$', '')
unemployment['Median HH Income'] = unemployment['Median HH Income']\
                                        .str.replace(',', '')                                        
unemployment['Median HH Income'] = pd.to_numeric(unemployment['Median HH Income'],
                                         errors='coerce')
#unemployment dataset has the word county following each county.
                                         
unemployment['County'] = unemployment['County'].str.replace('COUNTY', '')
unemployment['County'] = unemployment['County'].str.strip()
unemployment['County'] = unemployment['County'].apply(lambda x: x.upper())
education['Name'] = education['Name'].str.strip()
data['County'] = data['County'].str.strip()

#subset opoid overdose data for 2011-2016 total
overdose = data[['County', '2011-2016 Total']]

#calculate avg unemployment rate over the time period 2011-2015
unemployment['2011-2015 Avg'] = unemployment[['2011', '2012', '2013', '2014',
                                                '2015']].mean(axis=1)

merged = overdose.merge(education[['Name', '2011-2015' ]],
                    left_on='County', right_on='Name')
merged = merged.merge(unemployment[['County', '2011-2015 Avg', 'Median HH Income']])


merged.rename(columns={'2011-2016 Total': '2011-2016 Total Deaths', '2011-2015':
                      '2011-2015 Avg Edu', '2011-2015 Avg':'2011-2015 Avg Unemploy'}
                    , inplace=True)

merged.drop('Name', inplace=True, axis=1)

training = np.array(merged.iloc[:,1:5])

##Run PCA on training data. 
pca = PCA(n_components = 2)
pca.fit(training)

X = pca.transform(training)


##Run K-Means
kmean = KMeans(3, init='k-means++')
kmean.fit(X)

labels = kmean.labels_

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmean.predict(np.c_[xx.ravel(), yy.ravel()])

#obtain our predictions on the training
Z = kmean.predict(np.c_)



##plot time series of data ov
#opioid_df = data.iloc[0:88,0:12]
#opioid_pivot = pd.melt(opioid_df, id_vars = ['County'], var_name = 'Year', 
#               value_name = 'Count')
#
##get distinct counties
#counties = opioid_pivot['County'].unique().tolist()
#
#
#sns.tsplot(opioid_pivot, time='Year', value='Count', condition='County')
#
#
#
#for county in counties:
#    sns.kdeplot(opioid_dfly)
#    
#plt.plot(opioid_pivot['Year'], opioid_pivot['Count'], 
#                color=opioid_pivot['County'])




