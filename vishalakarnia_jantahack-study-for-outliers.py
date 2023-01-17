import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
janta_train=pd.read_csv("../input/train_Wc8LBpr.csv")

janta_train.head()
janta_train.info()
janta_train.isnull().sum()
janta_train.dtypes
import missingno as msno
msno.bar(janta_train, figsize=(10,5))

plt.show()
janta_train["Type_of_Cab"].unique()
janta_train["Type_of_Cab"].replace(np.nan,'?',inplace=True)
sns.countplot(data=janta_train,x='Type_of_Cab')
janta_train.loc[janta_train['Type_of_Cab']=='?'].head()
janta_train['Type_of_Cab'].value_counts()/len(janta_train)*100
sns.countplot(data=janta_train,x='Type_of_Cab',hue='Surge_Pricing_Type')
# janta_train.loc[(janta_train.Type_of_Cab=='?')&(janta_train.Surge_Pricing_Type=='1'),"Type_of_Cab"]="A"

# janta_train.loc[(janta_train.Type_of_Cab=='?')&(janta_train.Surge_Pricing_Type=='3'),"Type_of_Cab"]="D"
janta_train['Type_of_Cab'].replace('?',janta_train['Type_of_Cab'].mode()[0],inplace=True)
janta_train.isnull().sum()
janta_train['Customer_Since_Months'].replace(np.nan,janta_train['Customer_Since_Months'].mean(),inplace=True)

janta_train['Life_Style_Index'].replace(np.nan,janta_train['Life_Style_Index'].mean(),inplace=True)

janta_train['Confidence_Life_Style_Index'].replace(np.nan,janta_train['Confidence_Life_Style_Index'].mode()[0],inplace=True)

janta_train['Var1'].replace(np.nan,janta_train['Var1'].mean(),inplace=True)

janta_train.isnull().sum()
sns.heatmap(janta_train.isnull())

plt.figure(figsize=(100,100))

plt.show()
janta_train['Surge_Pricing_Type'].unique()
janta_train['Surge_Pricing_Type']=janta_train['Surge_Pricing_Type'].astype(object)
janta_train.dtypes
janat_train=janta_train.drop('Trip_ID',inplace=True,axis=1)
janta_train.dtypes
janta_train['Trip_Distance'].head()
janta_train['Trip_Distance']=janta_train['Trip_Distance'].round()
janta_train.head()
janta_train['Trip_Distance'].hist()
sns.countplot(data=janta_train,hue='Type_of_Cab',x='Trip_Distance')
sns.countplot(data=janta_train,hue='Surge_Pricing_Type',x='Destination_Type')
sns.countplot(data=janta_train,hue='Surge_Pricing_Type',x='Customer_Since_Months')
sns.boxplot(data=janta_train[['Trip_Distance','Var1']])
# q1=janta_train.quantile(0.25)

# q2=janta_train.quantile(0.75)

# iqr=q2-q1

# print(iqr)
# print(janta_train < (q1 - 1.5 * iqr)) |(janta_train > (q2 + 1.5 * iqr))
# from scipy import stats

# z = np.abs(stats.zscore(janta_train[['Trip_Distance','Customer_Since_Months','Life_Style_Index','Customer_Rating','Cancellation_Last_1Month'

#                                     ,'Var1','Var2','Var3']]))

# print(z)
# threshold = 3

# print(np.where(z > 3))
# print(z[4][1])
# janta_train= janta_train[(z < 3).all(axis=1)]
# janta_train.shape

# (124277, 13)
janta_train['Var1'].quantile([0.1,0.2,0.3,0.4])
janta_train['Var1'].quantile([0.97,0.98,0.99,1])
# janta_train.drop(janta_train[janta_train['Var1']<57.000000].index,axis=0,inplace=True)

janta_train.drop(janta_train[janta_train['Var1']>109.0].index,axis=0,inplace=True)
sns.boxplot(data=janta_train[['Trip_Distance','Var1']])
sns.boxplot(data=janta_train['Var2'])
janta_train['Var2'].quantile([0.1,0.2,0.3,0.4])
janta_train['Var2'].quantile([0.97,0.98,0.99,1])
janta_train.drop(janta_train[janta_train['Var2']>67.0].index,axis=0,inplace=True)
sns.boxplot(data=janta_train['Var2'])
sns.boxplot(data=janta_train['Trip_Distance'])
sns.boxplot(data=janta_train['Customer_Since_Months'])
sns.boxplot(data=janta_train['Life_Style_Index'])
janta_train['Life_Style_Index'].quantile([0.1,0.2,0.3,0.4])
janta_train['Life_Style_Index'].quantile([0.97,0.98,0.99,1])
janta_train.drop(janta_train[janta_train['Life_Style_Index']>3.353173].index,axis=0,inplace=True)
sns.boxplot(data=janta_train['Var3'])
janta_train['Var3'].quantile([0.97,0.98,0.99,1])
janta_train.drop(janta_train[janta_train['Var3']>107.0].index,axis=0,inplace=True)
janta_train.shape