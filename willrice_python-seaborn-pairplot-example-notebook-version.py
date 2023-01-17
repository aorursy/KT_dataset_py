# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#import

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64})
train['Age'] = train["Age"].fillna(train["Age"].median())

train["Survived"][train["Survived"]==1] = "Survived"

train["Survived"][train["Survived"]==0] = "Died"

train["ParentsAndChildren"] = train["Parch"]

train["SiblingsAndSpouses"] = train["SibSp"]

train = train[['Age','Fare','Survived','ParentsAndChildren','SiblingsAndSpouses','Pclass']]
plt.figure()

sns.pairplot(data=train, hue="Survived", dropna=True) 

plt.savefig("1_seaborn_pair_plot.png")