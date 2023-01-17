# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/percent-bachelors-degrees-women-usa.csv")
df.head()
df.info()
df.describe()
def group_by_5_years(value):

    if value < 1976 :

        return "71-76"

    elif value < 1981:

        return "76-81"

    elif value < 1986:

        return "81-86"

    elif value < 1991:

        return "86-91"

    elif value < 1996:

        return "91-96"

    elif value < 2001:

        return "96-01"

    elif value < 2006:

        return "01-06"

    else:

        return "06-11"
df.Year = df.Year.apply(group_by_5_years)
sns.barplot(x="Year",y="Agriculture",data = df)
group = df.groupby("Year").mean().reset_index()
def plotmatrix(start,end):

    fig, axs = plt.subplots(nrows = 2, ncols=2)

    i = 0

    cols = df.columns[start:end]

    fig.set_size_inches(14, 10)

    for indi in range(2):

        for indj in range(2):

                sns.barplot(x="Year",y=str(cols[i]),data = group,ax = axs[indi][indj],\

                            order = ['71-76', '76-81', '81-86', '86-91', '91-96','96-01','01-06','06-11'])

                i+=1

                #plt.xticks(rotation = 90)

plotmatrix(1,5)
plotmatrix(6,10)
plotmatrix(11,15)