# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/forest-fires-in-brazil/amazon.csv", encoding="latin1")
def count_plot(df):

    fig = plt.figure(figsize = (20, 3)) # width x height

    ax1 = fig.add_subplot(1, 3, 1) # row, column, position

    ax2 = fig.add_subplot(1, 3, 2)

    ax3 = fig.add_subplot(1, 3, 3)

    

    sns.countplot(data=df, x="state", ax=ax1)

    sns.countplot(data=df, x="year", ax=ax2)

    sns.countplot(data=df, x="month", ax=ax3)

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
count_plot(df)
month_dict = {'Janeiro':1, 'Fevereiro':2, 'Março':3, 'Abril':4, 'Maio':5, 'Junho':6, 'Julho':7,

       'Agosto':8, 'Setembro':9, 'Outubro':10, 'Novembro':11, 'Dezembro':12}

df["month"] = df["month"].map(month_dict)
def heatmap(df, index, ax):

    data = df.groupby(by=["state", index], axis=0)["number"].mean().unstack(level=0).reset_index()

    data = data.melt(id_vars=index, value_vars=df["state"].unique(), value_name="number")

    data = data.pivot(index, "state", "number")

    sns.heatmap(data, annot=False, fmt="f", ax=ax)

    plt.close(2)
fig = plt.figure(figsize = (20, 5)) # width x height

ax1 = fig.add_subplot(1, 2, 1) # row, column, position

ax2 = fig.add_subplot(1, 2, 2)

heatmap(df, "month", ax=ax1)

heatmap(df, "year", ax=ax2)
def region(state):

    north = ('Acre', 'Amapa', 'Amazonas', 'Rondonia', 'Roraima',

                                           'Tocantins', 'Pará')

    

    northeast = ('Bahia', 'Ceara', 'Maranhao', 'Sergipe', 'Paraiba',

                             'Pernambuco', 'Piau','Alagoas')

    

    centerwest = ('Goias', 'Mato Grosso', 'Distrito Federal')

    

    southeast = ('Espirito Santo', 'Rio', 'Sao Paulo', 'Minas Gerais')

    

    south = ('Santa Catarina')

    

    if state in north:

        return "north"

    if state in northeast:

        return "northeast"

    if state in centerwest:

        return "centerwest"

    if state in southeast:

        return "southeast"

    if state in south:

        return "south"

    

df["region"] = df["state"].apply(region)
def line_plot(df, index, ax):

    data = df.groupby(["region", index], axis=0)["number"].mean().unstack(level=0).reset_index().set_index(index)

    sns.lineplot(data=data, ax=ax, hue="region", legend="brief")

    ax.set(ylabel="number")

    ax.set(xlabel=index)

    ax.legend(loc=2)
fig = plt.figure(figsize = (20, 5)) # width x height

ax1 = fig.add_subplot(1, 2, 1) # row, column, position

ax2 = fig.add_subplot(1, 2, 2)

line_plot(df, "month", ax=ax1)

line_plot(df, "year", ax=ax2)