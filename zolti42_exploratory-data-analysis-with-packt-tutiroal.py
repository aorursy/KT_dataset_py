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



fname = "../input/train.csv"



data = pd.read_csv(fname)
len(data)
# Let's review the data structure!

data.head()
# Any missing values?

data.count()
# Let's review the extremes -> Add the column name within the brackets.

data["Age"].min(), data["Age"].max()
# How many passengers survived? .value_counts shows the population of the individual values.

data["Survived"].value_counts()
# We can express this in percentage as well.

data["Survived"].value_counts() * 100 / len(data)
# Let's visualize it

%matplotlib inline



alpha_color = 0.5



data["Survived"].value_counts().plot(kind="bar")
# And it can happen that sorting the values IS important.

data["Pclass"].value_counts().sort_index().plot(kind="bar",

                                               alpha=alpha_color)
# We can filter the data and then present only those who survived.

data[data["Survived"] == 1]["Age"].value_counts().sort_index().plot(kind="bar")
# Now we are going to make this more readable by bucketing (binning)

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]

data["AgeBin"] = pd.cut(data["Age"], bins)
# Let's see the binned data.

data[data["Survived"] == 1]["AgeBin"].value_counts().sort_index().plot(kind="bar")
# What about the dead?

data[data["Survived"] == 0]["AgeBin"].value_counts().sort_index().plot(kind="bar")
# How many people died in 1st class?

data[data["Pclass"] == 1]["Survived"].value_counts().plot(kind="bar")
# What about third class?

data[data["Pclass"] == 3]["Survived"].value_counts().plot(kind="bar")