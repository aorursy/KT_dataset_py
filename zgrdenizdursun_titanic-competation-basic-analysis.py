# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



from collections import Counter

import seaborn as sns

import warnings

warnings.filterwarnings

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
def bar_plot(variable):

    var = train_df[variable]

    varValue = var.value_counts()

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index,varValue, color = "#ffd571")

    plt.title(variable)

    plt.ylabel("Frequency")

    plt.xticks(varValue.index)

    

    
category1 = ["Survived","Sex","Pclass","SibSp","Parch","Embarked"]

for i in category1:

    bar_plot(i)
def hist_plot(var1):

    plt.figure(figsize = (9,3))

    plt.hist(train_df[var1] , bins = 50,color="#ff5f40")

    plt.xlabel(str(var1)+" "+"Values")

    plt.title(str(var1)+ " " + "Distrubition with histogram")
cat1 = ["Age","Fare","PassengerId"]

for i in cat1:

    hist_plot(i)
train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean()

#farklı olarak 'train_df.groupby("Pclass")[["Survived"]].mean()' şekilde de yazılabilir.
train_df.groupby("Sex")[["Survived"]].mean().sort_values(by="Survived",ascending=False)
train_df.groupby("SibSp")[["Survived"]].mean().sort_values(by="Survived",ascending=False)
train_df.groupby("Parch")[["Survived"]].mean().sort_values(by="Survived",ascending=False)
#def detect_outlier(df,features):

#    outlier_indeces = []

#    

#    for c in features:

#         

#        #Q1 (first quartile)

#        

#        Q1 = np.percentile(df[c],25)

#    

#        #Q3 (second quartile)

#        

#        Q3 = np.percentile(df[c],75)

#        

#        #IQR = Q3-Q1

#        

#        IQR = Q3-Q1

#        

#        #Outlier Step

#        

#        outlier_step = IQR* 1.5

#        

#        #detect ourlier and their indeces

#        

#        outlier_list_col = df[(df[c] < Q1 - outlier_step | df[c] > Q3 + outlier_step)].index

#        

#        #store indeces

#        

#        outlier_indeces.extend(outlier_list_col)

#        

#    outlier_indeces = Counter(outlier_indeces)

#    multiple_outliers = list(i for i,v in outlier_indeces.item() if v > 2)

#    

#    return multiple_outliers



    

   

    

    

    
traindf_len = len(train_df)

train_df = pd.concat([train_df,test_df], axis = 0).reset_index(drop = True)
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
#Null Embarked değerlerini bulma ve doldurma.

train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column = "Fare",by="Embarked");
train_df["Embarked"] = train_df["Embarked"].fillna("C")

train_df[train_df["Embarked"].isnull()]
#Boş Fare deperlerini bulma ve doldurma

train_df[train_df["Fare"].isnull()]
#Ayrıldığı Limanlara göre fiyat ortalaması

train_df.groupby("Embarked")[["Fare"]].mean()
#Sınıflarına göre fiyat ortalaması

np.mean(train_df[train_df["Pclass"] == 3]["Fare"])
# Sınıfı 3 olanların fiyat ortalaması ile doldurduk. 

train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))
train_df[train_df["Fare"].isnull()]
list1 = ["SibSp","Parch","Age","Fare","Survived"]

sns.heatmap(train_df[list1].corr(),annot=True,fmt=".2f");

#annot fonk. değerleri gösterir.

# fmt fonk. değerlerin virgülden sonra kaç basamak olduğunu gösterir.
g = sns.factorplot(x = "SibSp" , y = "Survived", data = train_df,kind="bar",size = 6);

g.set_ylabels("Survived Probablity") #y eksen başlığını güncellemek için set_ylabels("") kullanılır.

# sns.barplot(x = "SibSp" , y = "Survived", data = train_df); şeklinde de yazılabilir.

# sns.barplot() un extra metodlar yok. size gibi..

plt.show()
g = sns.factorplot(x = "Parch" , y = "Survived" ,data=train_df, kind = "bar" , size =6)

g.set_ylabels("Survived Probablity")

plt.show()

g = sns.factorplot(x = "Pclass" , y = "Survived" , data=train_df, kind="bar",size=6)

g.set_ylabels("Survived Probabilty")

plt.show()
g = sns.FacetGrid(train_df,col="Survived")

g.map(sns.distplot,"Age",bins=25,color="#ffa36c")

plt.show()
g = sns.FacetGrid(train_df,col="Survived",row="Pclass",size = 3)

g.map(plt.hist,"Age",bins=25,color="#8fc0a9")

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df,row = "Embarked", size = 3)

g.map(sns.pointplot,"Pclass","Survived","Sex")

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df,row = "Embarked" , col ="Survived",size = 3)

g.map(sns.barplot,"Sex","Fare")

g.add_legend()

plt.show()
train_df[train_df["Age"].isnull()]
sns.factorplot(x = "Sex",y = "Age", data = train_df, kind = "box")

plt.show()
sns.factorplot(x = "Sex" , y = "Age" , hue = "Pclass" , data = train_df , kind="box")

plt.show()
train_df.groupby("Pclass")[["Age"]].mean()
sns.factorplot(x = "Parch" , y = "Age" ,  data = train_df , kind="box")

sns.factorplot(x = "SibSp" , y = "Age" ,  data = train_df , kind="box")

plt.show()
train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]
sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(),annot=True)

plt.show()