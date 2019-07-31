# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:41:29 2019

@author: smedh
"""

import pandas as pd
from kmodes.kmodes import KModes

df = pd.read_csv (r'C:\Users\smedh\Documents\PGPBABI\Capstone\ws_python\Datasets\02_data_clean.csv')

df.describe()

df.columns
df.dtypes

"""
Age is the only numeric
"""

"""
On exploring we have a column called Illness that tags if the person has any one of the illness,
and f_illness for family history that can be taken for quick clustering.
If applicable, we can pick the specific illnesses later on to deep dive.

"""

df_v1 = df.drop(['Survey_ID','Heart_Condition', 'Cancer_Terminal', 'Diabetes', 'Lipid_Imbalance',
       'Immuno_Issues', 'Respiratory_Conditions', 'Liver_Kidney',
       'Other_Conditions', 'F_Heart_Condition',
       'F_Cancer_Terminal', 'F_Diabetes', 'F_Lipid_Imbalance',
       'F_Immuno_Issues', 'F_Respiratory_Conditions', 'F_Liver_Kidney',
       'F_Mental_Disorders', 'F_Other_Conditions'], inplace=False, axis=1) 

df_v1.dtypes

"""
Converting Age to categorical value for k-mode
On observibng histogram, 0-20 can be one and > 60 can be one group, but 20-60 
requires multiple group with spacing of 10
"""
df_v1.Age.hist()

custom_bucket_array = [0.,20., 30., 40., 50., 60.,100.]
custom_bucket_array   
    
Age_Group = pd.cut(df_v1['Age'], custom_bucket_array)   
Age_Group.describe()
Age_Group.value_counts()
        
df_v1['Age_Group'] = Age_Group       
df_v2=df_v1.drop(['Age'], inplace=False, axis=1) 

df_v2.dtypes

"""
Converting everything to object
"""

df_v2['Age_Group'] = df_v2['Age_Group'].astype('object')

"""
kmodes
"""

km = KModes(n_clusters=8, init='Huang', max_iter=50, n_init=5)

clusters = km.fit_predict(df_v2)

print(km.cluster_centroids_)

x = km.labels_
y = km.cluster_centroids_
y[0]

df_v3 = df_v2
df_v3['Cluster'] = x

df_v3.to_csv(r'C:\Users\smedh\Documents\PGPBABI\Capstone\ws_python\Datasets\03_kmode_result_new.csv', index=False)


"""

Smoker does not have the word yes in it and dissimilarity 
function might not be taking it as close match to smoker
So we ll pake it Yes, past Smoker instead of past smoker but not anymore
and observe if there are any differences

"""

mask = df_v3['Smoker'].str.contains('Past smoker, but not anymore', na=False)
df_v3.loc[mask, 'Smoker'] = 'Yes, past smoker'
df_v3=df_v3.drop(['Cluster'], inplace=False, axis=1) 



clusters = km.fit_predict(df_v3)

print(km.cluster_centroids_)

x = km.labels_

df_v4 = df_v3
df_v4['Cluster'] = x

y = df_v4['Cluster'].value_counts()


y = y.to_frame()
z=km.cluster_centroids_

col = ['Gender', 'Marital_Status', 'BMI', 'Children_Count', 'Smoker',
       'Existing_Health_Insurance', 'Physical_disability', 'Illness','F_Illness', 'Age_Group']

cluster_profile1 = pd.DataFrame(z, columns=col)
cluster_profile = cluster_profile1.merge(y, left_index=True, right_index=True)

df_v4.to_csv(r'C:\Users\smedh\Documents\PGPBABI\Capstone\ws_python\Datasets\03_kmode_result_fnl.csv', index=False)
cluster_profile.to_csv(r'C:\Users\smedh\Documents\PGPBABI\Capstone\ws_python\Datasets\04_profiles.csv', index=False)


