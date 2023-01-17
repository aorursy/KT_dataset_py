# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/data.csv')
df.head()
#On supprime la première colonne
df.drop(df.columns[[0]],axis=1,inplace=True)
df.head()
fig=plt.figure(figsize=(10,7))
sns.distplot(df.Age,color='teal');
#Qui sont les 5 plus vieux joueurs?
df.sort_values(by='Age',ascending=False).head()
#Note globale (overall) en fonction de l'age du joueur
sns.relplot(x='Age',y='Overall',data=df,kind='scatter',hue='Preferred Foot',height=8);
fig=plt.figure(figsize=(30,6));
sns.catplot(x='Work Rate',y='Overall',data=df,kind='violin');
df.groupby(by='Work Rate')['Work Rate'].count()
df.Value=df['Value'].apply(lambda x: x[1:])
df.Weight=df['Weight'].apply(lambda x: x[3:])
df.Wage=df['Wage'].apply(lambda x: x[1:])
df.Value=df['Value'].apply(lambda x: x[:-1])
df.Wage=df['Wage'].apply(lambda x: x[:-1])
df.Value=pd.to_numeric(df['Value'])
df.Wage=pd.to_numeric(df['Wage'])
fig=plt.figure(figsize=(10,8))
sns.scatterplot(x='Overall',y='Wage',data=df,hue='Preferred Foot')
#plt.xticks([1,100,200,300]);
fig=plt.figure(figsize=(10,8))
sns.scatterplot(x='Overall',y='Value',data=df)
sns.catplot(x='Position',y='Overall',data=df,kind='box',height=10);
sns.catplot(x='Nationality',y='Overall',data=df[df.Overall>90],kind='swarm',height=6)