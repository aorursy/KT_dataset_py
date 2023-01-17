#Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df=pd.read_csv("/kaggle/input/top50spotify2019/top50.csv",encoding='ISO-8859-1')
df.head()
df.info()
#Checking for Nulls
df.isnull().sum()
#Checking Valaue ranges
df.describe().T
#Distribution of Paopularity among the songs
plt.figure(figsize=(10,5))
sns.distplot(df['Popularity'],color='salmon')
#Comparing all Genres present in the dataset
sns.catplot(y = "Genre", kind = "count",
            palette = "plasma",size= 10
            , aspect = 0.6,
            data = df)
plt.show()
#Comparison of Genres basis the Popularity
plt.figure(figsize = (50,10))
sns.barplot(x ='Genre', y = 'Popularity', data = df)
#Artists who featured the most
df['Artist.Name'].value_counts().head(10)
#Top 10 Artist basis the popularity
artpop = df.groupby('Artist.Name')['Popularity'].sum()
artpop.sort_values(ascending=False, inplace=True)
top10 = artpop.iloc[:10]
sns.barplot(y = top10.index, x = top10.values, palette="plasma")
plt.show()
#Top 10 Tracks basis the popularity
spop = df.groupby('Track.Name')['Popularity'].sum()
spop.sort_values(ascending=False, inplace=True)
top10 = spop.iloc[:10]
sns.barplot(y = top10.index, x = top10.values, palette="plasma")
plt.show()
# Using correltion matrix to find out the maximum correlation between features through heatmaps
#Correlation for all variables
corrmat=df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, annot=True, square=True);
#We notice stong correlation between Beats per Minute & Speachiness, Energy & Valence, Engery & Loudness.
#We further look at the distribution pattern between the above three pairs
#Checking distribution between Beats per Minute & Speachiness
sns.jointplot(x=df['Beats.Per.Minute'], y=df['Speechiness.'], kind='reg',color='skyblue',height=7)
#Checking distribution between Energy & Valence
sns.jointplot(x=df['Energy'], y=df['Valence.'], kind='reg',color='violet',height=7)
#Checking distribution between Energy & Loudness
sns.jointplot(x=df['Energy'], y=df['Loudness..dB..'], kind='reg',color='red',height=7)