# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:07:09 2019

@author: smedh

K Means after 1 hot encoding
"""


import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

df = pd.read_csv (r'C:\Users\smedh\Documents\PGPBABI\Capstone\ws_python\Datasets\02_data_clean.csv')

df.describe()

df.columns
df.dtypes


"""
Age is the only numeric
"""

df_dummy1 = pd.get_dummies(df['Gender'])
df_dummy2 = pd.get_dummies(df['Marital_Status'])
df_dummy3 = pd.get_dummies(df['Smoker'])
df_dummy4 = pd.get_dummies(df['Existing_Health_Insurance'])

col_smoke = ['No','Yes_Past','Yes']
col_hi = ['No_Ins','Exp_Ins','Act_Ins']

df_dummy3.columns = col_smoke
df_dummy4.columns = col_hi

df_new = pd.concat([df, df_dummy1, df_dummy2, df_dummy3, df_dummy4], axis=1)
df_new = df_new.drop(['Gender','Marital_Status','Smoker','Existing_Health_Insurance'], inplace=False, axis=1)

"""
On exploring we have a column called Illness that tags if the person has any one of the illness,
and f_illness for family history that can be taken for quick clustering.
If applicable, we can pick the specific illnesses later on to deep dive.

"""

df_new1 = df_new.drop(['Heart_Condition', 'Cancer_Terminal', 'Diabetes', 'Lipid_Imbalance',
       'Immuno_Issues', 'Respiratory_Conditions', 'Liver_Kidney',
       'Other_Conditions', 'F_Heart_Condition',
       'F_Cancer_Terminal', 'F_Diabetes', 'F_Lipid_Imbalance',
       'F_Immuno_Issues', 'F_Respiratory_Conditions', 'F_Liver_Kidney',
       'F_Mental_Disorders', 'F_Other_Conditions'], inplace=False, axis=1) 

df_new1['Physical_disability_N'] = pd.Series(np.where(df_new1.Physical_disability.values == 'Yes', 1, 0),df_new1.index)
df_new1['Illness_N'] = pd.Series(np.where(df_new1.Illness.values == 'Y', 1, 0),df_new1.index)
df_new1['F_Illness_N'] = pd.Series(np.where(df_new1.F_Illness.values == 'Y', 1, 0),df_new1.index)

df_new1.dtypes

df_new1 = df_new1.drop([Physical_disability','Illness','F_Illness'], inplace=False, axis=1) 

df_new1['Children_N'] = pd.Series(np.where(df_new1.Children_Count.values == 'More than 2', 3, df_new1.Children_Count.values),df_new1.index)
df_new1['Children_N'] = df_new1['Children_N'].astype('int')

def calc_bmi (row):
   if row['BMI'] == '< 18.5 Underweight' :
      return 15.3
   if row['BMI'] == '18.5 - 24.9 Normal' :
      return 21.7
   if row['BMI'] == '25 - 29.9 Overweight' :
      return 27.4
   if row['BMI'] == '> 30 Obese' :
      return 32.5
   return 0

BMI_N = df_new1.apply (lambda row: calc_bmi (row), axis=1)
BMI_N = BMI_N.to_frame()

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(BMI_N)

df_new1['BMI_N'] = x_scaled

df_new2 = df_new1.drop(['BMI','Children_Count'], inplace=False, axis=1) 

df_new2.dtypes
df_new2 = df_new2.drop(['Survey_ID'], inplace=False, axis=1) 
df_norm = min_max_scaler.fit_transform(df_new2)

df_fnl = pd.DataFrame(data=df_norm,columns=df_new2.columns)

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df_fnl)
    Sum_of_squared_distances.append(km.inertia_)

import matplotlib.pyplot as plt

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

"""
Ignoring the only dip from 11 to 12, we are taking 8 as the optimal number of clusters

"""

print('NumPy covariance matrix: \n%s' %np.cov(df_fnl.T))

cov_mat = np.cov(df_fnl.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print(var_exp)
print(cum_var_exp)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    
    
from sklearn.decomposition import PCA # Principal Component Analysis module

pca = PCA(n_components=10)
df_10d = pca.fit_transform(df_fnl)    
    
kmeans = KMeans(n_clusters = 8)

X_clustered = kmeans.fit_predict(df_10d)    
    
df_fnl['X_cluster'] = X_clustered    
df['X_cluster'] = X_clustered   

df_temp = df_fnl[['Act_Ins','Illness_N','Yes']]
df_temp['Cluster'] = X_clustered
 
import seaborn as sns   
sns.pairplot(df_temp, hue='Cluster', palette= 'Dark2', diag_kind='kde',size=1.85)

from pandas.plotting import scatter_matrix
scatter_matrix(df_temp[['BMI_N','Cluster']], alpha=0.3, diagonal='kde')





