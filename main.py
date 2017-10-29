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
from sklearn.preprocessing import normalize

os.chdir("C:\\Users\\Chase\\Desktop\\School\\Fall 2017\\CMDA 4684\\County")

#read in data sources
data = pd.read_csv("data\ohio.csv")
unemployment = pd.read_csv("data\OhioUnemploymentReport.csv")
education = pd.read_csv("data\EducationReport.csv")

pop = pd.read_csv("data\popestimates.csv")


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

#clean population data
pop['Value'] = pop['Value'].str.replace(',', '')  
pop['Value'] =  pd.to_numeric(pop['Value'],
                                         errors='coerce')
pop.rename(columns={'Value':'Population'}, inplace=True)
pop['County'] = pop['County'].str.replace('COUNTY', '')
pop['County'] = pop['County'].str.strip()
pop['County'] = pop['County'].apply(lambda x: x.upper())



#calculate avg unemployment rate over the time period 2011-2015
unemployment['2011-2015 Avg'] = unemployment[['2011', '2012', '2013', '2014',
                                                '2015']].mean(axis=1)

#scale 2011-2015 avg per 10,000 people
merged = overdose.merge(education[['Name', '2011-2015' ]],
                    left_on='County', right_on='Name')
merged = merged.merge(pop, on='County')
merged = merged.merge(unemployment[['County', '2011-2015 Avg', 'Median HH Income']])


merged.rename(columns={'2011-2016 Total': '2011-2016 Total Deaths', '2011-2015':
                      '2011-2015 Avg Edu', '2011-2015 Avg':'2011-2015 Avg Unemploy'}
                    , inplace=True)

merged.drop('Name', inplace=True, axis=1)

#scale the deaths column by the population
merged['2011-2016 Total Deaths'] = merged['2011-2016 Total Deaths'] /(merged['Population']/100000)

merged.drop('Population', axis=1, inplace=True)
training = np.array(merged.iloc[:,2:5])

##we want to normalize this data column wise in the matrix.
training_norm = normalize(training, axis=0)

##Run PCA on training data. 
pca = PCA()
pca.fit(training_norm)

X = pca.transform(training_norm)


#Evaluate Kmeans model with varius ks

inertia_list = []
k_list  = []
for k in range(1, 10):
    kmean = KMeans(k, init='k-means++')
    kmean.fit(X)
    k_list.append(k)
    inertia_list.append(kmean.inertia_)
    


plt.plot(k_list, inertia_list)   
plt.title("Inertia(SSE) per Kmeans fit with given K. ")
plt.xlabel("K")
plt.ylabel("Inertia(SSE)")

##Run K-Means with selected k value from previous analysis. 
k=3
kmean = KMeans(k, init='k-means++')
kmean.fit(X)

labels = kmean.labels_
centroids = kmean.cluster_centers_
colors = ['b', 'g', 'y']

for i in range(k):
    xs = X[np.where(labels == i)]        
    plt.plot(xs[:, 0], xs[:, 1], 'o', color=colors[i], markersize=4)
    print(i)
    # plot the centroids
    lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    plt.setp(lines,ms=15.0)
    plt.setp(lines,mew=2.0)
    plt.title("Kmeans clustering with k=3")

#tie back kmeans results to original dataframe
merged['kmean_label'] = pd.Series(labels)

#if you want to export just the kmeans analysis
merged.to_csv("data_kmeans.csv")

mean_deaths = merged.groupby(['kmean_label'])['2011-2016 Total Deaths'].mean().reset_index(name="Mean Deaths")
stddev_deaths = merged.groupby(['kmean_label'])['2011-2016 Total Deaths'].std().reset_index(name="Std Dev Deaths")
mean_income = merged.groupby(['kmean_label'])['Median HH Income'].mean().reset_index(name="Mean Median HH Income")
stddev_income = merged.groupby(['kmean_label'])['Median HH Income'].std().reset_index(name="Std dev Median HH Income")
mean_education = merged.groupby(['kmean_label'])['2011-2015 Avg Edu'].mean().reset_index(name="Mean Education")
stddev_education = merged.groupby(['kmean_label'])['2011-2015 Avg Edu'].std().reset_index(name="Std dev Education")
mean_unemploy = merged.groupby(['kmean_label'])['2011-2015 Avg Unemploy'].mean().reset_index(name="Mean Unemployment")
stddev_unemploy = merged.groupby(['kmean_label'])['2011-2015 Avg Unemploy'].std().reset_index(name="Std dev Unemploy")

#get a dummy row count for the grouping
row_count = merged.groupby(['kmean_label'])['2011-2016 Total Deaths'].count().reset_index(name="n") 


grouped = mean_deaths.merge(stddev_deaths, on='kmean_label')
grouped = grouped.merge(mean_income, on='kmean_label')
grouped = grouped.merge(stddev_income, on='kmean_label')
grouped = grouped.merge(mean_education, on='kmean_label')
grouped = grouped.merge(stddev_education, on='kmean_label')
grouped = grouped.merge(mean_unemploy, on='kmean_label')
grouped = grouped.merge(stddev_unemploy, on='kmean_label')
grouped = grouped.merge(row_count, on='kmean_label')

#export the summary table
grouped.to_csv("kmeans_summary.csv")
