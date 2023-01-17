import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('../input/train.csv')
data.describe()
data.head()
data.isnull().sum()
data.shape
data=data.drop(columns=['Cabin', 'Name', 'Ticket'])
data.head()
data.describe()
data['Age'].dropna().plot.hist()
data['Age_types']=pd.cut(data['Age'], bins=[1, 14, 30, 50, 100], labels=["Child", "Adult", "MidAge", "Old"])
data['Age_types'].dropna().value_counts().plot.bar()
Age_types=data['Age_types'].dropna().value_counts()
Total_ppl=data['Age_types'].dropna().value_counts().sum()
(data['Age_types'].dropna().value_counts()/Total_ppl)*100
data.shape
Alive= data[data['Survived']==1]['Sex'].value_counts()
dead= data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([Alive,dead])
df.index = ['Alive','Dead']
df.plot(kind='bar',stacked=True, figsize=(12,10))
Alive_2= data[data['Survived']==1]['Age_types'].value_counts()
dead_2= data[data['Survived']==0]['Age_types'].value_counts()
df_2 = pd.DataFrame([Alive_2,dead_2])
df_2.index = ['Alive','Dead']
df_2.plot(kind='bar',stacked=True, figsize=(12,10))
Age_types
Alive_2
(Alive_2/Age_types)*100
total_alive=Alive_2.sum()
total_alive
(Alive_2/total_alive)*100
low_class= data[data['Pclass']==3]['Sex'].value_counts()
mid_class= data[data['Pclass']==2]['Sex'].value_counts()
high_class= data[data['Pclass']==1]['Sex'].value_counts()
df_3 = pd.DataFrame([low_class,mid_class, high_class])
df_3.index = ['Low','Middle', 'High']
df_3.plot(kind='bar',stacked=True, figsize=(12,10))
data['Fare'].dropna().plot.hist()
data['Fare_type']=pd.cut(data['Fare'], bins=[1, 50, 150, 550], labels=["Low_Fare", "Mid_Fare", "High_Fare"])
data['Fare_type'].dropna().value_counts().plot.bar()
data['Fare_type'].value_counts()
Total_fare=data['Fare_type'].value_counts().sum()
Total_fare
(data['Fare_type'].value_counts()/Total_fare)*100
Alive_3= data[data['Survived']==1]['Fare_type'].value_counts()
dead_3= data[data['Survived']==0]['Fare_type'].value_counts()
df_3 = pd.DataFrame([Alive_3,dead_3])
df_3.index = ['Alive','Dead']
df_3.plot(kind='bar',stacked=True, figsize=(12,10))