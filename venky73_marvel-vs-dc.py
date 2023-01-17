# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import regex as re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





marvel = pd.read_csv("../input/marvel-wikia-data.csv")

dc = pd.read_csv("../input/dc-wikia-data.csv")



# Any results you write to the current directory are saved as output.
print ("marvel\n",marvel.sample(3))

print ("DC\n",dc.sample(3))
print("Marvel:",marvel.shape," DC:",dc.shape)
marvel = marvel[marvel.APPEARANCES >= 100]

dc = dc[dc.APPEARANCES>=100]

print("Marvel:",len(marvel)," DC:",len(dc))
print("MARVEL:",marvel.columns)

print("DC:",dc.columns)
marvel.drop(columns=['ID','urlslug','GSM','FIRST APPEARANCE','Year'],inplace=True)

dc.drop(columns=['ID','urlslug','GSM','FIRST APPEARANCE','YEAR'],inplace=True)
print("MARVEL\n",marvel.isna().sum())

print("DC\n",dc.isna().sum())
print("Marvel\n",marvel.ALIGN.value_counts(dropna=False))

print("DC\n",dc.ALIGN.value_counts(dropna=False))
marvel.ALIGN.fillna(value = "Neutral Characters",inplace = True)

dc.ALIGN.fillna(value = "Neutral Characters",inplace = True)
print("MARVEL\n",marvel.ALIGN.isna().sum())

print("DC\n",dc.ALIGN.isna().sum())
print("MARVEL\n",marvel.EYE.value_counts(dropna=False))

print("DC\n",dc.EYE.value_counts(dropna=False))
eyes = ['Blue Eyes','Brown Eyes','Green Eyes','Red Eyes','Black Eyes']

eyes_after_marvel = []

for i in marvel.EYE.values:

    if i not in eyes:

        eyes_after_marvel.append('Different Eyes')

    else:

        eyes_after_marvel.append(i)

marvel['EYE'] = eyes_after_marvel

eyes_after_dc = []

for i in dc.EYE.values:

    if i not in eyes:

        eyes_after_dc.append('Different Eyes')

    else:

        eyes_after_dc.append(i)

dc['EYE'] = eyes_after_dc

print("MARVEL\n",marvel.EYE.value_counts(dropna=False))

print("DC\n",dc.EYE.value_counts(dropna=False))
print("Marvel\n",marvel.SEX.value_counts(dropna=False))

print("DC\n",dc.SEX.value_counts(dropna=False))
#let's take only Male,Female Characters Data

marvel = marvel[marvel.SEX.isin(["Male Characters","Female Characters"])]

dc = dc[dc.SEX.isin(["Male Characters","Female Characters"])]
print("Marvel\n",marvel.HAIR.value_counts(dropna=False))

print("DC\n",dc.HAIR.value_counts(dropna=False))
hair = ["Black Hair","Brown Hair","Blond Hair","Red Hair","Bald","No Hair","White Hair","Strawberry Blond Hair","Grey Hair","Auburn Hair"]

hair_after_marvel = []

for i in marvel.HAIR.values:

    if i not in hair:

        hair_after_marvel.append('Different Hair')

    else:

        hair_after_marvel.append(i)

marvel['HAIR'] = hair_after_marvel

hair_after_dc = []

for i in dc.HAIR.values:

    if i not in hair:

        hair_after_dc.append('Different Hair')

    else:

        hair_after_dc.append(i)

dc['HAIR'] = hair_after_dc

print("MARVEL\n",marvel.HAIR.value_counts(dropna=False))

print("DC\n",dc.HAIR.value_counts(dropna=False))
#so now, data doesn't have any missing values

print("MARVEL\n",marvel.isna().sum())

print("DC\n",dc.isna().sum())

print("GOOD Characters of MARVEL EYE color\n",marvel[marvel.ALIGN.isin(['Good Characters'])].EYE.value_counts())

print("GOOD Characters of DC EYE color\n",dc[dc.ALIGN.isin(['Good Characters'])].EYE.value_counts())
print("MARVEL\n",marvel.ALIGN.value_counts(normalize=True))

print("DC\n",dc.ALIGN.value_counts(normalize=True))

dc_vc = dc.ALIGN.value_counts(normalize=True).reset_index()

marvel_vc = marvel.ALIGN.value_counts(normalize=True).reset_index()

fig, axs = plt.subplots(nrows=2)

plt.subplots_adjust(hspace=0.5)

fig.set_size_inches(10, 8)

sns.barplot(x='index',y='ALIGN',data = dc_vc,ax=axs[0]).set_title('DC')

sns.barplot(x='index',y='ALIGN',data = marvel_vc,ax=axs[1]).set_title('MARVEL')
plt.figure(figsize=(10,5))

sns.countplot(x="ALIGN", data=marvel,hue = 'EYE').set_title("MARVEL")
plt.figure(figsize=(10,5))

sns.countplot(x="ALIGN", data=dc,hue = 'EYE').set_title("DC")
plt.figure(figsize=(10,5))

sns.countplot(x="ALIGN", data=marvel,hue = 'HAIR').set_title("MARVEL")
plt.figure(figsize=(10,5))

sns.countplot(x="ALIGN", data=dc,hue = 'HAIR').set_title("DC")
character_eyes = marvel.groupby(['ALIGN','EYE']).count().name.reset_index()

character_eyes = character_eyes.groupby(['ALIGN','EYE']).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()

character_eyes

plt.figure(figsize=(15,8))

sns.barplot(x="ALIGN",y='name', hue='EYE',data=character_eyes).set_title("MARVEL")
character_eyes = dc.groupby(['ALIGN','EYE']).count().name.reset_index()

character_eyes = character_eyes.groupby(['ALIGN','EYE']).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()

character_eyes

plt.figure(figsize=(15,8))

sns.barplot(x="ALIGN",y='name', hue='EYE',data=character_eyes).set_title("DC")
character_hair = marvel.groupby(['ALIGN','HAIR']).count().name.reset_index()

character_hair = character_hair.groupby(['ALIGN','HAIR']).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()

character_hair

plt.figure(figsize=(15,8))

sns.barplot(x="ALIGN",y='name', hue='HAIR',data=character_hair).set_title("MARVEL")
character_hair = dc.groupby(['ALIGN','HAIR']).count().name.reset_index()

character_hair = character_hair.groupby(['ALIGN','HAIR']).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()

character_hair

plt.figure(figsize=(15,8))

sns.barplot(x="ALIGN",y='name', hue='HAIR',data=character_hair).set_title("DC")
character_gender = marvel.groupby(['ALIGN','SEX']).count().name.reset_index()

character_gender = character_gender.groupby(['ALIGN','SEX']).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()

character_gender

plt.figure(figsize=(15,8))

sns.barplot(x="ALIGN",y='name', hue='SEX',data=character_gender).set_title("MARVEL")
character_gender = dc.groupby(['ALIGN','SEX']).count().name.reset_index()

character_gender = character_gender.groupby(['ALIGN','SEX']).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()

character_gender

plt.figure(figsize=(15,8))

sns.barplot(x="ALIGN",y='name', hue='SEX',data=character_gender).set_title("CD")
print("Top 10 most appearances of MARVEL\n",marvel.sort_values(by='APPEARANCES',ascending=False)[:10][['name','APPEARANCES']])

print("Top 10 most appearances of DC\n",dc.sort_values(by='APPEARANCES',ascending=False)[:11][['name','APPEARANCES']])
character_alive = marvel.groupby(['ALIGN',"ALIVE"]).count().name.reset_index()

character_alive = character_alive.groupby(['ALIGN',"ALIVE"]).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()

character_alive

plt.figure(figsize=(15,8))

sns.barplot(x="ALIGN",y='name', hue="ALIVE",data=character_alive).set_title("MARVEL")
character_alive = dc.groupby(['ALIGN',"ALIVE"]).count().name.reset_index()

character_alive = character_alive.groupby(['ALIGN',"ALIVE"]).sum().groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()

character_alive

plt.figure(figsize=(15,8))

sns.barplot(x="ALIGN",y='name', hue="ALIVE",data=character_alive).set_title("DC")