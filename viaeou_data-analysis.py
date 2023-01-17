# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# DESC_FILE = pd.read_excel('../input/education-statistics/EdStatsEXCEL.xlsx')
# print(DESC_FILE.shape,DESC_FILE.columns)
# DESC_FILE.head()
# country_series = pd.read_csv('../input/education-statistics/EdStatsCountry-Series.csv')
# country = pd.read_csv('../input/education-statistics/EdStatsCountry.csv')
# data = pd.read_csv('../input/education-statistics/EdStatsData.csv')
# foot_note = pd.read_csv('../input/education-statistics/EdStatsFootNote.csv')
# series = pd.read_csv('../input/education-statistics/EdStatsSeries.csv')
# country_series = country_series[country_series.columns[:-1]]
# print(country_series.shape,country_series.columns)
# country_series.head(10)
# country = country[country.columns[:-1]]
# print(country.shape,country.columns)
# country.head(10)
# data = data[data.columns[:-1]]
# print(data.shape,data.columns)
# data.head()
# foot_note = foot_note[foot_note.columns[:-1]]
# print(foot_note.shape,foot_note.columns)
# foot_note.head()
# series = series[series.columns[:-1]]
# print(series.shape,series.columns)
# series.head()
gapminder=pd.read_csv('../input/gapminder/gapminder.tsv', sep='\t')
gapminder[gapminder['country']=='China']
# gapminder.head()
df = gapminder[['country', 'year', 'pop']][gapminder['pop']>(gapminder['pop'].max()*0.2)]
df.head(20)
# gapminder.isnull().sum()
# gapminder.duplicated().sum()
# gapminder.drop_duplicates()
# gapminder.dropna()
# gapminder.sample(n=100)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(df, col="country", hue="country", palette="tab20c",
                     col_wrap=4, height=6)

# Draw a horizontal line to show the starting point
grid.map(plt.axhline, y=0, ls=":", c=".5")

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "year", "pop", marker="o")

# Adjust the tick positions and labels
grid.set(xticks=np.arange(12), yticks=[df['pop'].min(), df['pop'].max()],
         xlim=(1952, 2007), ylim=(df['pop'].min(), df['pop'].max()))

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)
# g = sns.relplot(x="year", y="pop", kind="line", data=df)
# g.fig.autofmt_xdate()
# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(df, col="walk", hue="walk", palette="tab20c",
                     col_wrap=4, height=1.5)

# Draw a horizontal line to show the starting point
grid.map(plt.axhline, y=0, ls=":", c=".5")

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "step", "position", marker="o")

# Adjust the tick positions and labels
grid.set(xticks=np.arange(5), yticks=[-3, 3],
         xlim=(-.5, 4.5), ylim=(-3.5, 3.5))

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)