# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:17:31 2019

@author: smedh
"""

"""

This code imports the survey responses and summarizes the columns and its behaviours
Also, we clean, remove redundant rows, treat missing values,
fix or delete the outliers/ invalid entries
The multiple choice survey questions are formatted into separate flag variables

Then we export the final dataset which is formatted, clean and workable for modeling

"""

import pandas as pd
import numpy as np

df = pd.read_excel (r'C:\Users\smedh\Documents\PGPBABI\Capstone\ws_python\google_data_survey_insurance1.xlsx')


#This gives all column names
list(df)

#This drop duplicate entries, if a person filled the survey twice with similar data
df.drop_duplicates(subset=None, keep='first', inplace=True)

df.head()

#We might not need name and timestamp for our immeadiate analysis - so we drop them

df_wip = df.drop(['Timestamp','Name'], inplace=False, axis=1)

#Generating Survey IDs to identify unique rows

df_wip.insert(0, 'Survey_ID', range(0, len(df)))
df_wip['Survey_ID'] = df_wip['Survey_ID'].astype(str)
df_wip['Survey_ID']=df_wip['Survey_ID'].apply(lambda x: '{0:0>4}'.format(x))
df_wip['Survey_ID'] = 'S' + df_wip['Survey_ID']

#Column names to a list
dfnames = df_wip.columns.tolist()
dfnames

#Modifying the names to readable simple format
dfnames[2]='Marital_Status'
dfnames[4]='BMI'
dfnames[5]='Resident'
dfnames[6]='Children_Count'
dfnames[7]='Smoker'
dfnames[8]='Existing_Health_Insurance'
dfnames[9]='Past_Claims'
dfnames[10]='Premium_Amount'
dfnames[11]='Physical_disability'
dfnames[12]='Self_Medical_History'
dfnames[13]='Family_Medical_History'

#Assigning the names to the dataset
df_wip.columns = dfnames
df_wip.dtypes

"""
From our survey structure, we know the fields Gender, Marital Status, BMI, Resident etc. are radio buttons
which are required fields. They do not need much validation and processing.

The only fields that need numeric validation is the Age and Premium Amount
The fields for Medical History is a multiple choice entry with a text option, 
and therefore will have to be converted to processable format
"""
Age = df_wip['Age']
max(Age)

min(Age)

df_wip['Age'] = df_wip['Age'].astype('float')
df_wip['Age'].values[df_wip['Age'] > 100] = np.nan
df_wip['Age'].values[df_wip['Age'] < 0] = np.nan

Age.describe()
df_wip['Age'].describe() 

df_wip['Age'].isnull().sum()
df_wip['Age'].isnull().sum() / len(df_wip)
# 0.011764705882352941 is too less of missing values - sp it is okay


"""

Age was a primary factor which we had to include in our analysis, 
so it has been replaced with null to avoid misleading results

However, the premium value is something we would use for prediction or pricing purposes,
therefore we shall replace with the mean or median for invalid value entries

"""
Premium = df_wip['Premium_Amount']
max(Premium)
min(Premium)

Premium.describe()

import seaborn as sns
sns.boxplot(x=Premium)

"""We observe that there is one extreme value which abruptly skews the whole plot 
so we replace with median"""

Premium.values[Premium == max(Premium)] = Premium.median()

#Then plot box plot again
Premium.describe()

import seaborn as sns
sns.boxplot(x=Premium)

#Now, we see a comprehensible plot with a lot of extreme outliers
#Trying to plot a scatter plot for the same


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df_wip['Premium_Amount'], df_wip['Age'])
plt.savefig(r'C:\Users\smedh\Documents\PGPBABI\Capstone\ws_python\foo.png')

#Calculating inter quartile range

Q1 = df_wip.quantile(0.25)
Q3 = df_wip.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

df_wip1=df_wip

"""
Right now, we shall replace all the outliers with nan, 
which could be treated or populated with projected entries during modeling
"""
df_wip1['Premium_Amount'].values[((df_wip1['Premium_Amount'] < (Q1[1] - 1.5 * IQR[1])) |(df_wip1['Premium_Amount'] > (Q3[1] + 1.5 * IQR[1])))]

df_wip1['Premium_Amount'].values[((df_wip1['Premium_Amount'] < (Q1[1] - 1.5 * IQR[1])) |(df_wip1['Premium_Amount'] > (Q3[1] + 1.5 * IQR[1])))]  = np.nan

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df_wip1['Premium_Amount'], df_wip1['Age'])
plt.savefig(r'C:\Users\smedh\Documents\PGPBABI\Capstone\ws_python\foo1.png')

df_wip1['Premium_Amount'].describe()


"""
Calculating the Medical History Flags from check box entries
"""

SelfHist = df_wip1['Self_Medical_History']
FamHist = df_wip1['Family_Medical_History']

SelfHist = SelfHist.to_frame()
FamHist = FamHist.to_frame()

#Convert all entries to lower case for better flagging
SelfHist['self_hist'] = SelfHist['Self_Medical_History'].str.lower()

#Replace all null entries to nan
SelfHist['self_hist'].replace(['none','nothing','na','nil','others','not applicable'], np.nan, inplace=True)

#Flag to imply whether the person has illness
SelfHist['Illness'] = np.where(SelfHist['self_hist'].isnull(), 'N', 'Y')

#Heart Condition
SelfHist['Heart_Condition'] = 'N'
mask = SelfHist['self_hist'].str.contains(r'heart condition', na=False)
SelfHist.loc[mask, 'Heart_Condition'] = 'Y'

#Cancer or Terminal Conditions
SelfHist['Cancer_Terminal'] = 'N'
mask = SelfHist['self_hist'].str.contains(r'cancer or terminal conditions', na=False)
SelfHist.loc[mask, 'Cancer_Terminal'] = 'Y'

#Diabetes
SelfHist['Diabetes'] = 'N'
mask = SelfHist['self_hist'].str.contains(r'diabetes', na=False)
SelfHist.loc[mask, 'Diabetes'] = 'Y'

#Lipid Imbalance
SelfHist['Lipid_Imbalance'] = 'N'
mask = SelfHist['self_hist'].str.contains(r'lipid imbalance', na=False)
SelfHist.loc[mask, 'Lipid_Imbalance'] = 'Y'

#HIV or Immuno disorders
SelfHist['Immuno_Issues'] = 'N'
mask = SelfHist['self_hist'].str.contains(r'immuno deficiency disorders', na=False)
SelfHist.loc[mask, 'Immuno_Issues'] = 'Y'

#Respiratory Conditions
SelfHist['Respiratory_Conditions'] = 'N'
mask = SelfHist['self_hist'].str.contains(r'respiratory conditions', na=False)
SelfHist.loc[mask, 'Respiratory_Conditions'] = 'Y'

#Liver or Kidney Conditions
SelfHist['Liver_Kidney'] = 'N'
mask = SelfHist['self_hist'].str.contains(r'kidney|liver', na=False)
SelfHist.loc[mask, 'Liver_Kidney'] = 'Y'

#Other Conditions
SelfHist['Other_Conditions'] = 'N'
mask = SelfHist['self_hist'].str.contains(r'brain|blood|thyroid|hormon|testicular|ovarian|gastro|nuero|bp', na=False)
SelfHist.loc[mask, 'Other_Conditions'] = 'Y'

"""
Similarly for Family History
"""

#Convert all entries to lower case for better flagging
FamHist['fam_hist'] = FamHist['Family_Medical_History'].str.lower()

#Replace all null entries to nan
FamHist['fam_hist'].replace(['none','nothing','na','nil','others','not applicable'], np.nan, inplace=True)
FamHist.fam_hist.unique()
#Flag to imply whether the person has illness
FamHist['Illness'] = np.where(FamHist['fam_hist'].isnull(), 'N', 'Y')

#Heart Condition
FamHist['Heart_Condition'] = 'N'
mask = FamHist['fam_hist'].str.contains(r'heart condition', na=False)
FamHist.loc[mask, 'Heart_Condition'] = 'Y'

#Cancer or Terminal Conditions
FamHist['Cancer_Terminal'] = 'N'
mask = FamHist['fam_hist'].str.contains(r'cancer or terminal conditions', na=False)
FamHist.loc[mask, 'Cancer_Terminal'] = 'Y'

#Diabetes
FamHist['Diabetes'] = 'N'
mask = FamHist['fam_hist'].str.contains(r'diabetes', na=False)
FamHist.loc[mask, 'Diabetes'] = 'Y'

#Lipid Imbalance
FamHist['Lipid_Imbalance'] = 'N'
mask = FamHist['fam_hist'].str.contains(r'lipid imbalance', na=False)
FamHist.loc[mask, 'Lipid_Imbalance'] = 'Y'

#HIV or Immuno disorders
FamHist['Immuno_Issues'] = 'N'
mask = FamHist['fam_hist'].str.contains(r'immuno deficiency disorders', na=False)
FamHist.loc[mask, 'Immuno_Issues'] = 'Y'

#Respiratory Conditions
FamHist['Respiratory_Conditions'] = 'N'
mask = FamHist['fam_hist'].str.contains(r'respiratory|asthma|bronchitis', na=False)
FamHist.loc[mask, 'Respiratory_Conditions'] = 'Y'

#Liver or Kidney Conditions
FamHist['Liver_Kidney'] = 'N'
mask = FamHist['fam_hist'].str.contains(r'kidney|liver', na=False)
FamHist.loc[mask, 'Liver_Kidney'] = 'Y'

#Liver or Kidney Conditions
FamHist['Mental_Disorders'] = 'N'
mask = FamHist['fam_hist'].str.contains(r'mental', na=False)
FamHist.loc[mask, 'Mental_Disorders'] = 'Y'

#Other Conditions
FamHist['Other_Conditions'] = 'N'
mask = FamHist['fam_hist'].str.contains(r'brain|blood|thyroid|hormon|testicular|ovarian|gastro|nuero|bp', na=False)
FamHist.loc[mask, 'Other_Conditions'] = 'Y'

"""
Adding the values back to the original data frame as flags, 
after dropping the multi chouice fields
"""
#To add a F_ Prefix to all columns to identify Family History
FamHist.columns = 'F_' + FamHist.columns

#To pick required fields alone
SelfHist1 = SelfHist.drop(['Self_Medical_History','self_hist'], inplace=False, axis=1)
FamHist1 = FamHist.drop(['F_Family_Medical_History','F_fam_hist'], inplace=False, axis=1)

df_wip2 = df_wip1.iloc[:, 0:12]

df_final = pd.concat([df_wip2, SelfHist1, FamHist1], axis=1)

finalcolumns = df_final.columns.tolist()
finalcolumns


df_final['Premium_Amount'].describe()

#To write the dataset onto csv

df_final.to_csv(r'C:\Users\smedh\Documents\PGPBABI\Capstone\ws_python\Datasets\01_data_final_v2.csv', index=False)


fig, ax = plt.subplots()
df_final['Age'].value_counts().plot(ax=ax, kind='bar')


df_final.Age.hist()

df_final.describe()
df_final.describe(include='all')


