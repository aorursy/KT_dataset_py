# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.plotly as py

import plotly.graph_objs as go



# read crashes

df = pd.read_csv("../input/Airplane_Crashes_and_Fatalities_Since_1908.csv",

sep=",",

encoding = "ISO-8859-1",

header=0)

# Any results you write to the current directory are saved as output.
# date time grooming

# we are not interested in minutes

# bin time by hour

df["Hour"] = pd.to_numeric(df["Time"].str.extract(r"^ *(\d+)\:\d+ *$", expand=False))

df.drop(columns=["Time"], inplace=True)



# not interested in month day, interested in year and month

df["Month"] = pd.to_numeric(df["Date"].str.extract(r"^ *(\d+)\/\d+/\d+ *$", expand=False))

df["Year"] = pd.to_numeric(df["Date"].str.extract(r"^ *\d+\/\d+/(\d+) *$", expand=False))

df.drop(columns=["Date"], inplace=True)



df.head()
# Now let us have a look at the NaN values

df.isna().sum()
# ...and how the data is distributed

df.describe()
sns.catplot(x="Year", kind="count", palette="ch:.25", data=df);



# create 10 year intervals

df["Year10"] = df["Year"] // 10 * 10



# show years on X and frequencies on Y

sns.catplot(x="Year10", kind="count", palette="ch:.25", data=df);

# create correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# draw heatmap

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, mask=mask)
sns.relplot(x="Year", y="Aboard", data=df)
df[(df["Fatalities"] >= 300)]
# compute survivors

df["Survivors"] = df["Aboard"] - df["Fatalities"]



# show heatmap

# create correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# draw heatmap

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, mask=mask)
sns.relplot(x="Aboard", y="Fatalities", data=df)

sns.relplot(x="Aboard", y="Survivors", data=df)
df["Survival Rate"] = df["Survivors"] / df["Aboard"]



sns.relplot(x="Year", y="Survival Rate", data=df)
df2 = df.query("Year >= 1950 and Aboard > 10")

sns.relplot(x="Year", y="Survival Rate", data=df2)





sns.boxplot(x="Year10", y="Survival Rate", data=df2)
pp_df = df2[["Year", "Survival Rate"]]



sns.pairplot(pp_df)