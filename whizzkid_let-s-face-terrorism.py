import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
df=pd.read_csv("../input/globalterrorismdb_0718dist.csv",encoding='ISO-8859-1',

              usecols=['iyear','extended','country','country_txt','region','region_txt',

                      'provstate','city','success','attacktype1','attacktype1_txt',

                      'targtype1','targtype1_txt','nkill','nwound','weapdetail'])

# extended=whether attack was extended for more than 24hour
df.sample(2)
plt.figure(figsize=(10,10))

sns.barplot(df.iyear.value_counts().index,df.iyear.value_counts().values)
sns.barplot(df.extended.value_counts().index,df.extended.value_counts().values)

# 0 means < 24hours

# 1 means > 24hours
#top 50 countries with high attacks

country_with_high_attacks=df.country_txt.value_counts().sort_values(ascending=False)[:50].index 

df_country_with_high_attacks=df[df.country_txt.isin(country_with_high_attacks)]
plt.figure(figsize=(15,15))

sns.barplot(df_country_with_high_attacks.country_txt.value_counts().index,

            df_country_with_high_attacks.country_txt.value_counts().values)

plt.xticks(range(df_country_with_high_attacks.country_txt.nunique()),rotation='vertical')

plt.show()
plt.figure(figsize=(10,10))

sns.barplot(df.attacktype1_txt.value_counts().index,df.attacktype1_txt.value_counts().values)

plt.xticks(range(df.attacktype1_txt.nunique()),rotation='vertical')

plt.show()
df_count=df.success.value_counts()

sns.barplot(df_count.index,df_count.values)
plt.figure(figsize=(10,10))

df_count=df.targtype1_txt.value_counts()

sns.barplot(df_count.index,df_count.values)

plt.xticks(range(df.targtype1_txt.nunique()),rotation='vertical')

plt.show()