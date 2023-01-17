# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(r"/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv")
df.head()

df.tail()
df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
df['Date_Time'] = pd.to_datetime(df['Datum']) #converting format to date and time



df['Year'] = df['Date_Time'].apply(lambda datetime: datetime.year) #extracting the years for launch



df["Country"] = df["Location"].apply(lambda location: location.split(", ")[-1]) #extracting countries for launch
df.head()
df = df.drop(columns=['Location', 'Datum'])
fig_dims = (20, 25)

fig, ax = plt.subplots(figsize=fig_dims)

ax.set(xscale="log")

sns.countplot(y="Company Name",data=df, order = df["Company Name"].value_counts().index).set_title('Company v/s Launches')
fig_dims = (15, 20)

fig, ax = plt.subplots(figsize=fig_dims)

sns.countplot(y="Country",data=df, order = df["Country"].value_counts().index).set_title('Country v/s Launches')
fig_dims = (15, 20)

fig, ax = plt.subplots(figsize=fig_dims)

sns.countplot(y="Year",data=df).set_title('Year v/s Launches')
fig_dims = (10,10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.countplot(x="Status Mission",data=df).set_title('Mission Status')
fig_dims = (10,10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.countplot(x="Status Rocket",data=df).set_title('Rocket Status')