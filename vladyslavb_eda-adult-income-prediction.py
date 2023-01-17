# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
df = pd.read_csv('../input/adult.csv')
df.head()
df.describe().T
df.info()
sns.boxplot(df['age'])
sns.boxplot(df['hours-per-week'])
sns.boxplot(df['capital-gain'])
sns.boxplot(df['educational-num'])
plt.figure(figsize=(25,50))

sns.countplot(y='age', hue='income', data=df)
df.groupby('income')['age'].mean()
df.groupby('income')['age'].std()
df[df['income']=='>50K']['education'].value_counts(normalize=True)
df[df['gender']=='Male'].groupby('relationship')['income'].value_counts(normalize=True)
df['hours-per-week'].max()
df[(df['hours-per-week']==df['hours-per-week'].max())]
df[(df['hours-per-week']==df['hours-per-week'].max())].info()
df[(df['hours-per-week']==df['hours-per-week'].max()) & (df['income']=='>50K')]
df[(df['hours-per-week']==df['hours-per-week'].max()) & (df['income']=='>50K')].info()
sns.heatmap(df[['capital-gain','hours-per-week']].corr(method='spearman'));
from scipy.stats import spearmanr, kendalltau

spearmanr(df['age'], df['hours-per-week'])
kendalltau(df['age'], df['hours-per-week'])
V=df.drop_duplicates(subset=['education'])['education']

V.unique()
df['education-num']=df['education'].map({'11th':6, 'HS-grad':8, 'Assoc-acdm':10, 'Some-college':12, '10th':5,

       'Prof-school':9, '7th-8th':3, 'Bachelors':13, 'Masters':14, 'Doctorate':15,

       '5th-6th':2, 'Assoc-voc':11, '9th':4, '12th':7, '1st-4th':1, 'Preschool':0})

df['education-num']
plt.figure(figsize=(25,100))

sns.countplot(y='hours-per-week', hue='education-num', data=df[df['native-country']!='United-States'])
sns.jointplot(x='education-num', y='hours-per-week', data=df);
from scipy.stats import spearmanr, kendalltau

spearmanr(df['hours-per-week'], df['education-num'])
kendalltau(df['hours-per-week'], df['education-num'])
df.groupby('income')['native-country'].value_counts(normalize=True)
df.groupby('native-country')['income'].value_counts(normalize=True)
pd.crosstab(df['native-country'], df['income'])
plt.figure(figsize=(15,8))

sns.countplot(y='native-country', hue='income', data=df[df['native-country']!='United-States'])

#Сша провизуализируем отдельно, так как на общем графике, их результаты слишком уменьшают масштаб остальных (из-за большого количества представителей в данной выборке)
sns.countplot(y='native-country', hue='income', data=df[df['native-country']=='United-States'])
V=df.drop_duplicates(subset=['native-country'])['native-country']

df['native-country-num']=df['native-country'].map({V.iloc[i] : i  for i in range(V.size)})

df['native-country-num']
df.head()
df['income-num']=df['income'].map({'<=50K':1, '>50K':0})
from scipy.stats import pointbiserialr

pointbiserialr(df['income-num'], df['native-country-num'])
df.groupby('native-country')['educational-num'].mean()
plt.figure(figsize=(25,100))

sns.countplot(y='native-country', hue='educational-num', data=df[df['native-country']!='United-States'])
sns.countplot(y='native-country', hue='educational-num', data=df[df['native-country']=='United-States'])
from scipy.stats import pointbiserialr

pointbiserialr(df['educational-num'], df['native-country-num'])
pd.crosstab(df['gender'], df['income'])
from statsmodels.graphics.mosaicplot import mosaic

mosaic(df, ['gender','income'])
from scipy.stats import chi2_contingency, fisher_exact

df['gender-num']=df['gender'].map({'Male':0, 'Female':1})
chi2_contingency(pd.crosstab(df['gender-num'], df['income-num']))
fisher_exact(pd.crosstab(df['gender-num'], df['income-num']))
pd.crosstab(df['race'], df['income'])
from statsmodels.graphics.mosaicplot import mosaic

mosaic(df, ['race','income'])
V=df.drop_duplicates(subset=['race'])['race']

df['race-num']=df['race'].map({V.iloc[i] : i for i in range(V.size)})

V.unique()
df['race-num']
chi2_contingency(pd.crosstab(df['race-num'], df['income-num']))
plt.figure(figsize=(25,150))

sns.countplot(y='age', hue='educational-num', data=df[df['native-country']!='United-States'])
from scipy.stats import spearmanr, kendalltau

spearmanr(df['age'], df['educational-num'])
kendalltau(df['age'], df['educational-num'])
sns.heatmap(df[['age','hours-per-week']].corr(method='spearman'));
from scipy.stats import spearmanr, kendalltau

spearmanr(df['age'], df['hours-per-week'])
kendalltau(df['age'], df['hours-per-week'])