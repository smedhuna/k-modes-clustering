# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 18:34:23 2019

@author: smedh
"""
import pandas as pd

df = pd.read_csv (r'C:\Users\smedh\Documents\PGPBABI\Capstone\ws_python\Datasets\01_data_final_v2.csv')

df.head()

res = df.columns[df.isnull().any()]
print(res)

df['Age'].isnull().sum()
df['Resident'].isnull().sum()

#['Age', 'Resident', 'Past_Claims', 'Premium_Amount']

"""
We do not require Resident for the analysis, because it is irrelevant.

Similarly the columns for past claimns or premium amount is only required for our post 
analysis validation or optimization. We however require the existing insurance column for our plan C

Age has 4 missing values which has to be treated
"""

df_v1 = df.drop(['Resident', 'Past_Claims', 'Premium_Amount'], inplace=False, axis=1)

res = df_v1.columns[df_v1.isnull().any()]
print(res)


df_v1['Age'].describe()

"""

We can observe that the mean and median age are around the same range of 30-33, 
therefore let us pick the median age to maintain whole number and 
to ensure we do not get outlier impact

"""

df_v1['Age'].values[df_v1['Age'].isnull()] = df_v1['Age'].median()

df_v1['Age'].isnull().sum()

res = df_v1.columns[df_v1.isnull().any()]
print(res)

#No more nulls
df_v1.to_csv(r'C:\Users\smedh\Documents\PGPBABI\Capstone\ws_python\Datasets\02_data_clean.csv', index=False)

