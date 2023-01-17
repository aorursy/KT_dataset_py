# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/googleplaystore.csv')
df.head()
df.tail()
df.describe()
df.dropna(subset=["Content Rating"],inplace=True)
df.drop(["Current Ver","Android Ver"],axis=1,inplace=True)
df.info()
df.Category.unique()
df.Type.unique()
df.Type.value_counts()
df.Type.fillna('Free',inplace=True)
df.Rating.value_counts().head(4)
df.Rating.value_counts().head(10).plot(kind='barh')
df.groupby("Category").Rating.mean()
df.groupby("Category").Rating.median()
df.Rating = df.Rating.fillna(df.groupby("Category").Rating.transform('median'))
df.info()
df.Price.value_counts().head(3)
df.Price = df.Price.str.replace("$","")

df.Price.value_counts().head(3)
df.Size.value_counts().head(3)
df.Size.unique()
df.Size = df.Size.str.replace("Varies with device", 'NaN')

df.Size = df.Size.str.replace("M", "e6")

df.Size = df.Size.str.replace("k", "e3")
df.Size = df.Size.astype(float)
df.dropna(subset=['Size'],inplace=True)
df.info()
df.Installs.value_counts().head(3)

df.Installs.unique()
df.Installs = df.Installs.str.replace("+","")

df.Installs = df.Installs.str.replace(",","")
df.Installs.value_counts()
df.Installs = df.Installs.astype(float)
df.sample(10)
df.Installs.value_counts()
df["Last Updated"] = pd.to_datetime(df["Last Updated"])

df.Category = df.Category.astype("category")

df.Reviews = df.Reviews.astype(float)

df.Type = df.Type.astype("category")

df.Price = df.Price.astype(float)

df["Content Rating"] = df["Content Rating"].astype("category")

df.Genres = df.Genres.astype("category")

df.info()
df.columns = df.columns.str.replace(" ","_")



sns.distplot(df.Rating,kde=False, rug=True)
fig = plt.figure(figsize=(18,18))

sns.countplot(x='Rating',hue='Type',data=df,dodge=True)

plt.show()
f, ax = plt.subplots(figsize=(12,12))

sns.scatterplot(data=df,x="Reviews",y="Rating",hue="Type")

f, ax = plt.subplots(figsize=(12,12))

sns.scatterplot(data=df,x="Reviews",y="Rating",hue="Type")

plt.xlim(0,10**6)



plt.show()

f, ax = plt.subplots(figsize=(12,12))

sns.scatterplot(data=df,x="Reviews",y="Rating",hue="Type",size='Installs')



plt.show()

fig, ax = plt.subplots(figsize=(12,12))

plt.scatter('Reviews','Rating',data=df)

plt.show()
f, ax = plt.subplots(figsize=(10,10))

g =sns.regplot(data=df,x="Reviews",y="Installs",fit_reg=False,x_jitter=.1)

g.set(xscale='log',yscale='log')

plt.show()

f, ax = plt.subplots(figsize=(10,10))

sns.countplot(x=df.loc[(df.Type == "Paid") & (df.Price<5),"Price"])

plt.xticks(rotation=60)
f, ax = plt.subplots(figsize=(10,10))

sns.countplot(x=df.loc[(df.Type == "Paid") & (df.Price >5) & (df.Price<10) ,"Price"])

plt.xticks(rotation=60)
f, ax = plt.subplots(figsize=(10,10))

g =sns.regplot(x=df.loc[df.Type == "Paid","Price"],y=df.loc[df.Type == "Paid","Rating"],fit_reg=False)



plt.show()

f, ax = plt.subplots(figsize=(10,10))

g =sns.regplot(x=df.loc[(df.Type == "Paid")&(df.Price<50),"Price"],y=df.loc[(df.Type == "Paid")&(df.Price<50),"Rating"],fit_reg=False)



plt.show()

df.Category.value_counts().head(4)
df.Category.value_counts().head(4)
sns.regplot(data=df, x="Size", y="Rating",scatter_kws={'alpha':0.3})
sns.countplot(x="Content_Rating",data=df)
sns.countplot(x=df.Rating.loc[df.Content_Rating == "Everyone"],data=df)

plt.xticks(rotation=60)
sns.catplot(x="Rating",y="Category", data=df)