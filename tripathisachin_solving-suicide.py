# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#For visualization
import matplotlib.pyplot as plt
import seaborn as sns 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input"))
df=pd.read_csv("../input/Suicides in India 2001-2012.csv") #Creating a DF of the csv file
df.head(10)
#Correcting the name of the columns
df.replace('A & N Islands', 'A & N Islands (Ut)', inplace=True)
df.replace('Chandigarh', 'Chandigarh (Ut)', inplace=True)
df.replace('D & N Haveli', 'D & N Haveli (Ut)', inplace=True)
df.replace('Daman & Diu', 'Daman & Diu (Ut)', inplace=True)
df.replace('Lakshadweep', 'Lakshadweep (Ut)', inplace=True)
df.replace('Puducherry', 'Puducherry (Ut)', inplace=True)

df.replace('Bankruptcy or Sudden change in Economic', 'Bankruptcy or Sudden change in Economic Status', inplace=True)
df.replace('By Other means (please specify)', 'Other', inplace=True)
df.replace('Other Causes (Please Specity)', 'Other', inplace=True)
df.replace('Others (Please Specify)', 'Other', inplace=True)
df.replace('Not having Children(Barrenness/Impotency', 'Not having Children (Barrenness/Impotency', inplace=True)
#Finding different categories of 'Type' present in the data
pd.unique(df['Type_code'])
#Grouping them by the 'Type' present
df_a=df.groupby('Type_code')
#Getting df by each value of 'Type_code'
def gtgrp(x):
    dfs=df_a.get_group(x)
    return dfs
#storing them in separate dfs
cause=gtgrp('Causes')
edu=gtgrp('Education_Status')
means=gtgrp('Means_adopted')
professional=gtgrp('Professional_Profile')
social=gtgrp('Social_Status')
#to check the distribution across the Type and across 'Gender' and 'Age-group'
def pvtcol(x):
    dfg=pd.pivot_table(df, values = 'Total', index='Gender', columns = (x.Type))#gender
    dfa=pd.pivot_table(df, values = 'Total', index='Age_group', columns = (x.Type))#age
    dfg['max']=(dfg.idxmax(axis=1))#col with max value in a row
    dfa['max']=(dfa.idxmax(axis=1))
    return(dfg,dfa)
#to check the distribution across the Type and across 'State'
def pvtcol_a(x):
    dfg=pd.pivot_table(df, values = 'Total', index='State', columns = (x.Type))#gender
    dfg['max']=(dfg.idxmax(axis=1))#col with max value in a row
    
    return(dfg)
gender_causes,age_causes=pvtcol(cause)
gender_edu,age_edu=pvtcol(edu)
gender_means,age_means=pvtcol(means)
gender_prof,age_prof=pvtcol(professional)
gender_social,age_social=pvtcol(social)
state_causes=pvtcol_a(cause)
state_edu=pvtcol_a(edu)
state_means=pvtcol_a(means)
state_prof=pvtcol_a(professional)
state_social=pvtcol_a(social)
#calculating the sum across the row dropping the categorical value
gender_causes['sum'] = gender_causes.drop('max', axis=1).sum(axis=1)
gender_causes.plot.bar(figsize=(30,20))
plt.xlabel('Gender', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.suptitle('Causes-Gender',fontsize=30)
plt.xticks(fontsize=16, rotation=360)
plt.yticks(fontsize=16)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=18)
age_causes.plot.bar(figsize=(50,30))
plt.xlabel('Age', fontsize=25)
plt.ylabel('Frequency', fontsize=20)
plt.suptitle('Causes-Age',fontsize=32)
plt.xticks(fontsize=25, rotation=360)
plt.yticks(fontsize=20)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=30)
gender_edu.plot.bar(figsize=(30,20))
plt.xlabel('Gender', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.suptitle('Education-Gender',fontsize=30)
plt.xticks(fontsize=16, rotation=360)
plt.yticks(fontsize=16)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=20)
age_means.plot.bar(figsize=(50,30))
plt.xlabel('Age', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.suptitle('Means-Age',fontsize=32)
plt.xticks(fontsize=18, rotation=360)
plt.yticks(fontsize=18)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=30)

gender_prof.plot.bar(figsize=(30,20))
plt.xlabel('Gender', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.suptitle('Prof-Gender',fontsize=30)
plt.xticks(fontsize=16, rotation=360)
plt.yticks(fontsize=16)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=18)

gender_social.plot.bar(figsize=(30,20))
plt.xlabel('Gender', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.suptitle('Social-Gender',fontsize=30)
plt.xticks(fontsize=16, rotation=360)
plt.yticks(fontsize=16)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=18)
state_causes['sum'] = state_causes.drop('max', axis=1).sum(axis=1)
(state_causes.head())
(state_causes.drop('sum', axis=1)).head().plot.bar(figsize=(50,30))
plt.xlabel('Age', fontsize=25)
plt.ylabel('Frequency', fontsize=20)
plt.suptitle('Causes-State',fontsize=32)
plt.xticks(fontsize=25, rotation=360)
plt.yticks(fontsize=20)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=30)
state_prof.head().plot.bar(figsize=(50,30))
plt.xlabel('Age', fontsize=25)
plt.ylabel('Frequency', fontsize=20)
plt.suptitle('Causes-Age',fontsize=32)
plt.xticks(fontsize=25, rotation=360)
plt.yticks(fontsize=20)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=30)
state_edu.head().plot.bar(figsize=(50,30))
plt.xlabel('Age', fontsize=25)
plt.ylabel('Frequency', fontsize=20)
plt.suptitle('Causes-Age',fontsize=32)
plt.xticks(fontsize=25, rotation=360)
plt.yticks(fontsize=20)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=30)