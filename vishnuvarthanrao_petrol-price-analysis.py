# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
path =''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path =os.path.join(dirname, filename)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(path)

df.head(5)
mean_df = df.mean()
mean_df
def plot_func(mean_df,title):
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(aspect="equal"))

    wedges, texts, autotexts = ax.pie(mean_df, autopct=lambda pct: func(pct),
                                      textprops=dict(color="w"))

    ax.legend(wedges, ["Delhi","Kolkata","Mumbai","Chennai"],
              title=title,
              loc="center",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=20, weight="bold")

    ax.set_title(title)
    plt.show()
plot_func(mean_df ,"Distribution of Mean")
def return_corresponding_year_df(year):
    return df[df["Date"].str.contains(year)]
df_2014 = return_corresponding_year_df('2014')
df_2014
mean_2014 = df_2014.mean()
mean_2014
plot_func(mean_2014,"Distribution of Mean 2014")
df_2014.describe()
df_2015 = return_corresponding_year_df('2015')
df_2015
mean_2015 = df_2015.mean()
plot_func(mean_2015,"Distibution of Mean 2015")
df_2015.describe()
