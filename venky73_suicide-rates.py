# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/who_suicide_statistics.csv")
df.sample(3)
df.info()
df.isna().sum()
df.describe()
df.head()
df.shape
df = df[~df.population.isna()]
df.isna().sum()
df = df[~df.suicides_no.isna()]
df.isna().sum()
df.country.unique()
df.year.unique()
df = df[df.year > 2006]
df["percent_ratio"] =  100*df.suicides_no/df.population
df.head()
sns.barplot(x = "sex", y = "percent_ratio", data = df, hue = "age").set_title("Gender vs Suicides Ratio")
sns.barplot(x = "year", y = "percent_ratio", data = df).set_title("Years vs Suicide rates")
df[(df.year>2011)].sort_values(by = ["percent_ratio"], ascending = False)[:10]
sns.barplot(x = "country", y = "percent_ratio", data = df[(df.year>2011)&(df.percent_ratio > 0.08)]).set_title("Country vs Suicides Ratio")

plt.xticks(rotation = 90)