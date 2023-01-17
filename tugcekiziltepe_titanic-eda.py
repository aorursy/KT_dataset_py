# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

#plt.style.available



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"]
train_df.columns
train_df.head()
train_df.describe()
train_df.info()
def plot_bar_graph(variable):

    """

    input: variable ex: "Sex" 

    output: bar plot & value count

    

    """

    #get feature

    var= train_df[variable]

    

    # count number of categorical variable

    varValue = var.value_counts()

    

    #visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable, varValue))
category1 = ["Survived","Sex", "Pclass","Embarked", "SibSp", "Parch"]

for i in category1:

    plot_bar_graph(i)
category2 = ["Cabin", "Name" , "Ticket"]

for i in category2:

    print("{} \n".format(train_df[i].value_counts()))

    
def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(train_df[variable], bins= 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} Distribution with Histogram Graph".format(variable))

    plt.show()
numericVar = ["Fare", "Age", "PassengerId"]

for i in numericVar:

    plot_hist(i)
# Pclass vs Survived

train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
# Sex vs Survived

train_df[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
# SibSp vs Survived

train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
# Parch vs Survived

train_df[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
# Cabin vs Survived

train_df[["Cabin", "Survived"]].groupby(["Cabin"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
def detect_outliers(df, features):

    outlier_indices = []

    

    for i in features:

        # 1st quartile

        Q1 = np.percentile(df[i], 25)

        

        # 3rd quartile

        Q3 = np.percentile(df[i], 75)

        

        #IQR

        IQR = Q3 - Q1

        

        #Outlier Step

        outlier_step = IQR * 1.5

        

        #Detect outlier and their indices

        outlier_list_col = df[(df[i] < Q1 - outlier_step) | (df[i] > Q3 + outlier_step)].index

        

        # Store indices

        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v>2)

    return multiple_outliers
train_df.loc[detect_outliers(train_df, ["Age", "SibSp", "Parch", "Fare"])]
#drop outliers

train_df = train_df.drop(detect_outliers(train_df, ["Age", "SibSp", "Parch", "Fare"]),axis = 0).reset_index(drop = True)
train_df_len = len(train_df)

train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)
train_df.head()
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column= "Fare", by = "Embarked")

plt.show()
train_df["Embarked"] = train_df["Embarked"].fillna("C")

train_df[train_df["Embarked"].isnull()]
train_df[train_df["Fare"].isnull()]
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))