import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
import xgboost as xgb
from scipy import stats
campus = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv', index_col='sl_no')
campus.head()
campus.info()
campus.shape
campus.isnull().sum()
sns.countplot(campus['status'])
campus.columns
catcols = ['gender', 'ssc_b', 'hsc_s','hsc_b', 'degree_t', 'workex', 'specialisation']
numcols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'salary']
plt.figure(figsize=(15,20))
for i in range(0,7):
  plt.subplot(4,3,i+1)
  sns.countplot(x = catcols[i], hue = 'status', data = campus)
plt.show()
plt.figure(figsize=(20,20))
for i in range(0,6):
  plt.subplot(4,3,i+1)
  sns.barplot(y = numcols[i], x = 'status', data = campus)
plt.show()
plt.figure(figsize=(20,20))
for i in range(0,6):
  plt.subplot(4,3,i+1)
  sns.distplot(campus[numcols[i]])
plt.show()
sns.barplot(campus['gender'], campus['salary'])
plt.figure(figsize=(10,5))
sns.heatmap(campus.corr(), annot=True)
plt.figure(figsize=(20,20))
for i in range(0,6):
  plt.subplot(4,3,i+1)
  sns.boxplot(campus[numcols[i]])
plt.show()
avg_pct = []
for i in range(1, 216):
  a = round((campus['ssc_p'][i]+campus['hsc_p'][i]+campus['degree_p'][i]+campus['etest_p'][i]+campus['mba_p'][i])/5, 2)
  avg_pct.append(a)
campus['total_p'] = avg_pct

campus.head()
sns.barplot(x = campus['status'], y= campus['total_p']) # those who are having higher total average percentage are likely to placed more 
sns.countplot(x = 'specialisation', hue = 'status', data = campus) #Specialization in marketing and finance are much demanded by corporate
sns.pairplot(campus, hue = 'status')
campus['salary'].fillna(0, inplace = True)
campus.head()