import numpy as np # linear algebra 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 

import matplotlib.pyplot as plt

import seaborn as sns 

df = pd.read_csv("/kaggle/input/windows-store/msft.csv")
df.head()
df.tail()
df.dropna()
fig_dims = (10, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.countplot(y="Category",data=df, order=df.Category.value_counts().index).set_title('Number of Apps per Category') 
fig_dims = (25, 15)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x=df["Category"],y=df["No of people Rated"]).set_title('Number of Ratings per Category') 
df.loc[~df["Price"].isin(['Free']),"Price"] = "Not Free" #replacing all Paid values
fig_dims = (20, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.countplot(x="Price", data=df).set_title('Free vs Not Free')
fig_dims = (30, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.countplot(x="Category", hue = "Price", data=df).set_title('Free vs Not Free per Category')
fig_dims = (20, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.countplot(x="Rating", data=df).set_title('OVERALL RATINGS')