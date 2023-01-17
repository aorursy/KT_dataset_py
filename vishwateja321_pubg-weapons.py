# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file = '/kaggle/input/pubg-weapon-stats/pubg-weapon-stats.csv'
df = pd.read_csv(file)
df.head(44)
df.shape
df.info()
df.describe(include='all').transpose()
df.isna().sum()
df = df.fillna(0)
df.isna().sum()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,5))
df.groupby(['Weapon Type'])['Damage'].mean().plot()
pd.pivot_table(df,['Damage','BDMG_0','BDMG_1','BDMG_2','BDMG_3','HDMG_0','HDMG_1','HDMG_2','HDMG_3','Shots to Kill (Chest)','Shots to Kill (Head)'],index=['Weapon Type','Weapon Name'])
wp = pd.pivot_table(df,['Bullet Speed','Rate of Fire','Range','Damage Per Second'],index=['Weapon Type','Fire Mode','Weapon Name'])
wp
wp.sort_values(by=['Range','Bullet Speed'],ascending=False)

df.head()
sns.distplot(df['Damage'])
plt.figure(figsize=(15,5))
sns.barplot(df['Weapon Type'],df['Damage'])
plt.figure(figsize=(15,5))
sns.barplot(df['Weapon Type'],df['BDMG_3'])
plt.figure(figsize=(15,5))
sns.barplot(df['Weapon Type'],df['HDMG_3'])
plt.figure(figsize=(20,5))
df.groupby(['Weapon Type'])['Shots to Kill (Head)'].median().plot()
plt.figure(figsize=(20,5))
df.groupby(['Weapon Type'])['Shots to Kill (Chest)'].median().plot()
plt.figure(figsize=(25,5))
sns.boxplot(df['Weapon Type'],df['Damage'])
plt.figure(figsize=(25,5))
sns.boxplot(df['Weapon Type'],df['Range'])
plt.figure(figsize=(25,5))
sns.pointplot(df['Weapon Type'],df['Bullet Speed'])
df.groupby(['Weapon Type','Bullet Type'])['Damage'].mean()