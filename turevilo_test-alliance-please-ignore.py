# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/vgsales.csv")

df = df[df["Year"] <=2017]

#df.info()
histogramm = df.Year.plot.hist(grid = True, rwidth = 0.9).grid(alpha=0.75, dashes=(7,7))

histogramm
df[df["Year"] < 2018].plot.scatter("Year", "Global_Sales", alpha=0.5, grid = True).grid(alpha=0.75, dashes=(7,7))
PuGlo=df.groupby("Publisher")["Global_Sales"].sum().reset_index().sort_values("Global_Sales",ascending=False)

PuGlo.index = pd.RangeIndex(1, len(PuGlo)+1)

PuGlo.head(20)