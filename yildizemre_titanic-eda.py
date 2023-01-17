# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"]
train_df.columns

train_df.head()

train_df.tail()
train_df.describe()
train_df.info()
def bar_plot(variable):

    """

        input: variable ex: "Sex"

        output: bar plot & value count

    """

    # get feature

    var = train_df[variable]

    # count number of categorical variable(value/sample)

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))

    
category1 = ["Survived","Sex","Pclass","Embarked","SibSp", "Parch"]

for c in category1:

    bar_plot(c)
category2 = ["Cabin","Name","Ticket"]

for c in category2:

    print("{} \n ".format(train_df[c].value_counts()))
def plot_hist(veriable):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[veriable])

    plt.xlabel(veriable)

    plt.ylabel("Degeler")

    plt.title("{}".format(veriable))
numericVar = ["Fare", "Age","PassengerId"]

for c in numericVar:

    plot_hist(c)
train_df[["Pclass","Survived"]]
#Pclass - Survived

train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index = False).mean()
train_df[["Pclass","Survived"]].groupby(["Pclass"] , as_index = False).mean().sort_values(by = "Survived", ascending = False)
#Sex - Survided

train_df[["Sex","Survived"]].groupby(["Sex"] , as_index = False ).mean().sort_values(by= "Survived", ascending=False)
#* SibSp - Survided

train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by="Survived", ascending=False)


#* Parch - Survived

train_df[["Parch","Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(by="Survived",ascending = False)
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # detect outlier and their indeces

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        # store indeces

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers
#for example

a = ["a","a","a","b","b","b"]

Counter(a)
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
train_df.head()
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column="Fare", by="Embarked")

plt.show()
train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]
train_df[train_df["Fare"].isnull()]
train_df["Fare"]=train_df["Fare"].fillna(train_df[train_df["Pclass"]==3]["Fare"]).mean()
train_df[train_df["Fare"].isnull()]