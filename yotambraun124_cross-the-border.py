# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import date

from datetime import time

from datetime import datetime



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/us-border-crossing-data/Border_Crossing_Entry_Data.csv")

df.head()
len(df)
df.isnull().sum()
df.info()
df.keys()
df["Border"].unique()
df["Port Name"].unique()
df["Value"].mean()
biggest_five_value = df["Value"].nlargest(n=5)

biggest_five_value.index
df.loc[biggest_five_value.index]
us_mexico_border = df[df["Border"]=="US-Mexico Border"]
us_mexico_border.head()
us_canada_border =df[df["Border"]=="US-Canada Border"]
us_canada_border.head()
m = sns.countplot(x='Measure',hue="State", data=us_mexico_border)

m.set_xticklabels(m.get_xticklabels(),rotation=90)
c = sns.countplot(x='Measure',hue="State", data=us_canada_border)

c.set_xticklabels(c.get_xticklabels(),rotation=90)
us_mexico_border["date"] = us_mexico_border["Date"].astype('datetime64[ns]')

us_mexico_border.date.dt.year.head()
us_mexico_border.date.dt.month.head()
df_group_by_year = us_mexico_border.groupby(us_mexico_border.date.dt.year).mean()["Value"]

df_group_by_year.plot.bar()
df_group_by_type_m = us_mexico_border.groupby(us_mexico_border.Measure).mean()["Value"]

df_group_by_type_c = us_canada_border.groupby(us_canada_border.Measure).mean()["Value"]

df_group_by_type_m.plot.bar(color="r").legend(["mexico","canada"])

df_group_by_type_c.plot.bar(color="b").legend(["mexico","canada"])

df_group_by_Port_Name_m = us_mexico_border.groupby(us_mexico_border["Port Name"]).mean()["Value"]



df_group_by_Port_Name.plot.bar()
fig,axes = plt.subplots()

df_group_by_Port_Name_c = us_canada_border.groupby(us_canada_border["Port Name"]).mean()["Value"]



df_group_by_Port_Name_c.plot.bar(fig=0.001)

len(df_group_by_Port_Name_c.index)
