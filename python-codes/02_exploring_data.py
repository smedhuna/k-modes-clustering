# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:48:11 2019

@author: smedh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set()
plt.rcParams['patch.force_edgecolor'] = True
plt.rcParams["axes.labelweight"] = "bold"
import warnings
warnings.filterwarnings("ignore")

df_final.info() #Base Data
# Data Head and Data Description
df_final.head()
df_final.describe() #analysis on nemeric data
# Data Visualization
from pandas.plotting import scatter_matrix
scatter_matrix(df_final[['Age','Premium_Amount']], alpha=0.3, diagonal='kde')
plt.figure(1)
df_final.groupby(['Gender'])['Age'].count().plot.bar()
plt.subplot(2,2,2)
df_final.groupby(['BMI'])['Age'].count().plot.bar()
plt.subplot(2,2,3)
df_final.groupby(['Marital_Status'])['Premium_Amount'].mean().plot.bar()

plt.figure(2)
plt.subplot(2,2,1)
data.groupby(['sex'])['bmi'].sum().plot.bar()
plt.subplot(2,2,2)
data.groupby(['smoker'])['bmi'].sum().plot.bar()
plt.subplot(2,2,3)
data.groupby(['region'])['bmi'].sum().plot.bar()
# Check for Null Data
data.isnull().sum()