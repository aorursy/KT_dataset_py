#   Processing

import pandas as pd

import numpy as np

import re

#   Visuals

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize']=(20,10)
df = pd.read_csv("../input/inc_occ_gender.csv", na_values="Na")
df.head()
df.count()
df.isnull().sum()
df = df.dropna().reset_index(drop=True)

df.isnull().sum()
df.head()
titles = []

indexes = []

for i in range(df.shape[0]):

    if re.findall('([A-Z][A-Z]+)', df['Occupation'][i]): #  Groups are in capital letters

        titles.append(' '.join(re.findall('([A-Z][A-Z]+)', df['Occupation'][i])))

        indexes.append(i)
titles
df.head()
dfTitles = df.loc[df['Occupation'].isin(titles)].reset_index(drop=True)

dfTitles.head()
ax = sns.barplot(x="M_weekly", y="Occupation", data=dfTitles.sort_values('M_weekly',ascending=False))

_ = ax.set_xlim(0, 2000)
ax = sns.barplot(x="F_weekly", y="Occupation", data=dfTitles.sort_values('F_weekly',ascending=False))

_ = ax.set_xlim(0, 2000)
dfTitles['diff'] = dfTitles['M_weekly']-dfTitles['F_weekly']

dfTitles.head()
ax = sns.barplot(x="diff", y="Occupation", data=dfTitles.sort_values('diff',ascending=False))
dfsubTitles = df.loc[~df['Occupation'].isin(titles)].reset_index(drop=True)

dfsubTitles.head()
df.loc[1]
dfsubTitles['diff'] = dfsubTitles['M_weekly'] - dfsubTitles['F_weekly']

dfsubTitles.head()
ax = sns.barplot(x="diff", y="Occupation", data=dfsubTitles.sort_values('diff',ascending=False)[:15])
ax = sns.barplot(x="diff", y="Occupation", data=dfsubTitles.sort_values('diff',ascending=False)[-10:])
ax = sns.distplot(dfsubTitles['M_weekly'],bins=20, color='blue',label='Male')

ax = sns.distplot(dfsubTitles['F_weekly'], bins=20, color='pink',label='Female')

ax.set(ylim=(0, 0.002))

plt.plot([dfsubTitles['M_weekly'].median(),dfsubTitles['M_weekly'].median()],[0, 0.0004], linewidth=2, color='blue',label='Male median')

plt.plot([dfsubTitles['F_weekly'].median(),dfsubTitles['F_weekly'].median()],[0, 0.0004], linewidth=2, color='red',label='Female median')

_=ax.legend()
sns.boxplot(data=df[['M_weekly','F_weekly']])
print("**************MALE**************")

print(df['M_weekly'].describe())

print("*************FEMALE*************")

print(df['F_weekly'].describe())