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



# Any results you write to the current directory are saved as output.
titanic = pd.read_csv("/kaggle/input/titanic-cleaned-data/train_clean.csv")
titanic.head()
titanic.shape
titanic.columns
titanic.groupby("Survived").size()

# can see that the sum of people who survived and didn't survive equal the total number of rows (891)
titanic.groupby("Survived").size()/titanic.shape[0]*100
titanic["Age"].max()
titanic["Fare"].median()
titanic.sort_values("Fare",ascending=False).iloc[0:10]
total = titanic.groupby("Pclass").size()

total
survived = titanic.groupby("Pclass").sum()["Survived"]

survived
survived/total*100
titanic.groupby("Sex").sum()["Survived"]/titanic.groupby("Sex").size()*100
# number of female in 1st class that survived

fsurvived = titanic[titanic["Sex"] == "female"].query("Pclass == 1").sum()["Survived"]

# number of femaes in 1st class

ftotal = titanic[titanic["Sex"] == "female"].query("Pclass == 1").shape[0]

fsurvived/ftotal*100
titanic.groupby("Survived").mean()["Age"]
titanic.Age.plot.hist()