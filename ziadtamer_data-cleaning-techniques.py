# Import Libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Load Data
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

 #Concatenate train & test
train_objs_num = len(train)
y = train['Survived']
dataset = pd.concat(objs=[train.drop(columns=['Survived']), test], axis=0)
train.head()
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_data.head()
plt.figure(figsize=(15,8))
sns.distplot(dataset.Pclass, bins =30)
dropRows = dataset
dropRows.dropna(inplace = True)
dropRows.isnull().sum()
df = dataset
df.dropna(how = 'all',inplace = True)
df.isnull().sum()
df = dataset
df.dropna(axis = 1,inplace = True)
df = dataset
df.Age.fillna(-1,inplace=True)
df.Age.isnull().sum()
df = train
df.Age.isnull().sum()
mean = df.Age.mean()
mean
df.Age.replace(np.NaN, mean).head(10)
df = train
df.Age.fillna(df.Age.median(),inplace=True)
df.head()
data=train
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.head()
df = train
df.Cabin.head(10)
df.Cabin.fillna('Unknown').head(10)
import pandas_profiling 
train.profile_report()