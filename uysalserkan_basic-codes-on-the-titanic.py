# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Modelling importations
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


plt.style.use("seaborn-whitegrid")

import seaborn as sns
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
train_df.describe()
train_df.info()
def bar_plot(variable):
    """
        input: variable e.g. Sex
        output: bar plot & value count
    """
    # get feature
    var = train_df[variable]
    #count numer of categorical variable(value/sample)
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
category1 = ["Survived","Sex","SibSp","Parch","Pclass","Embarked"]
for cat in category1:
    bar_plot(cat)
category2 = ["Cabin","Name","Ticket"]
for cat in category2:
    print("{} \n".format(train_df[cat].value_counts))
def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable],bins=75)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with histogram".format(variable))
    plt.show()
numericVariables = ["Fare","Age","PassengerId"]
for numer in numericVariables:
    plot_hist(numer)
#Pclass - Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False)
#Sex - Survived
train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False)
#SibSp - Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)
#Parch - Survived
train_df[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived",ascending=False)
def detect_outlier(df,features):
    outlier_indices = []
    
    
    for c in features:
        #first quartile
        Q1 = np.percentile(df[c],25)
        #third quartile
        Q3 = np.percentile(df[c],75)
        #IQR
        IQR = Q3 - Q1 
        
        #Outlier Step
        outlier_step = IQR * 1.5
        
        #detect outlier and their indices
        outlier_list_column = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        #store indices
        outlier_indices.extend(outlier_list_column)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
train_df.loc[detect_outlier(train_df,["Age","SibSp","Fare","Parch"])]
# Drop outliers
train_df = train_df.drop(detect_outlier(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=False)
train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column="Fare",by="Embarked")
plt.show()
train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Fare"].isnull()]
train_df[train_df["Pclass"]==3]["Fare"].mean()
train_df["Fare"] = train_df["Fare"].fillna(train_df[train_df["Pclass"]==3]["Fare"].mean())
list1 = ["SibSp","Parch","Age","Fare","Survived"]
sns.heatmap(train_df[list1].corr(),annot=True,fmt=".2f")
g = sns.factorplot(x="SibSp",y="Survived",data=train_df,kind="bar",size=5)
g.set_ylabels("Survived Probability")
plt.show()
g = sns.factorplot(x="Parch",y="Survived",kind="bar",data=train_df,size=5)
g.set_ylabels("Survived Probability")
plt.show()
g = sns.factorplot(x="Pclass",y="Survived",data=train_df,kind="bar",size=5)
g.set_ylabels("Survived Probability")
plt.show()
g = sns.FacetGrid(train_df,col="Survived")
g.map(sns.distplot,"Age",bins=25)
plt.show()
g = sns.FacetGrid(train_df,col="Survived",row="Pclass")
g.map(plt.hist,"Age",bins=25)
g.add_legend()
plt.show()
g = sns.FacetGrid(train_df,row="Embarked")
g.map(sns.pointplot,"Pclass","Survived","Sex",bins=25)
g.add_legend()
plt.show()
g = sns.FacetGrid(train_df,row="Embarked",col="Survived",size=3.5)
g.map(sns.barplot,"Sex","Fare")
g.add_legend()
plt.show()
train_df[train_df["Age"].isnull()]
sns.factorplot(x="Sex",y="Age",data=train_df,kind="box")
plt.show()
sns.factorplot(x="Sex",y="Age",hue="Pclass",data=train_df,kind="box")
plt.show()
sns.factorplot(x="Parch",y="Age",data=train_df,kind="box")
sns.factorplot(x="SibSp",y="Age",data=train_df,kind="box")
plt.show()
train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]
sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(),annot=True)
plt.show()
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_df["Age"][((train_df["SibSp"]==train_df.iloc[i]["SibSp"]) & (train_df["Parch"]==train_df.iloc[i]["Parch"]) & (train_df["Pclass"]==train_df.iloc[i]["Pclass"]))].median()
    age_med = train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_med
train_df["Name"].sample(10)
name = train_df["Name"]
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
train_df["Title"].sample(10)
sns.countplot(x="Title",data=train_df)
plt.xticks(rotation=90)
plt.show()
train_df["Title"] = train_df["Title"].replace(["Lady","The Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i=="Mile" or i=="Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]
sns.countplot(x="Title",data=train_df)
plt.xticks(rotation=90)
plt.show()
train_df["Title"].sample(10)
g = sns.factorplot(x="Title",y="Survived",data=train_df,kind="bar")
g.set_xticklabels(["Master","Mrs","Mr","Other"])
g.set_ylabels("Survival Probability")
plt.show()
train_df.drop(labels = ["Name"],axis=1,inplace=True)
train_df.sample(10)
train_df = pd.get_dummies(train_df,columns=["Title"])
train_df.head()
train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1
g = sns.factorplot(x="Fsize",y="Survived",data=train_df,kind="bar")
g.set_ylabels("Survival")
plt.show()
train_df["family_size"] = [1 if i<5 else 0 for i in train_df["Fsize"]]
train_df.sample(10)
sns.countplot(x="family_size",data=train_df)
plt.show()
g = sns.factorplot(x="family_size",y="Survived",data=train_df,kind="bar")
g.set_ylabels("Survival")
plt.show()
train_df = pd.get_dummies(train_df,columns=["family_size"])
train_df.sample(10)
train_df["Embarked"].sample(10)
sns.countplot(x="Embarked",data=train_df)
plt.show()
train_df = pd.get_dummies(train_df,columns=["Embarked"])
train_df.sample(10)
train_df["Ticket"].sample(10)
tickets = []
for i in list(train_df["Ticket"]):
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        tickets.append("x")
train_df["Ticket"] = tickets
train_df["Ticket"].sample(10)
train_df = pd.get_dummies(train_df,columns=["Ticket"],prefix="T")
train_df.sample(10)
sns.countplot(x="Pclass",data=train_df)
plt.show()
train_df["Pclass"] = train_df["Pclass"].astype("category")
train_df = pd.get_dummies(train_df,columns=["Pclass"])
train_df.sample(10)
train_df["Sex"] = train_df["Sex"].astype("category")
train_df = pd.get_dummies(train_df,columns=["Sex"])
train_df.sample(10)
train_df.drop(labels=["PassengerId","Cabin"],axis=1,inplace=True)
train_df_len
test = train_df[train_df_len:]
test.drop(labels=["Survived"],axis=1,inplace=True)
test.sample(7)
train=train_df[:train_df_len]
x_train=train.drop(labels = "Survived",axis=1)
y_train=train["Survived"]
x_train,x_test,y_train,y_test = train_test_split(x_train, y_train, test_size = 0.3,random_state = 25)
logReg = LogisticRegression()
logReg.fit(x_train,y_train)

accLogRegTrain = logReg.score(x_train,y_train)
accLogRegTest = logReg.score(x_test,y_test)
print("Train Acc: {}\nTest Acc: {}".format(accLogRegTrain,accLogRegTest))
random_state = 25
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]
cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i],param_grid=classifier_param[i],cv = StratifiedKFold(n_splits=10),scoring="accuracy",n_jobs=-1,verbose=1)
    clf.fit(x_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "soft", n_jobs = -1)
votingC = votingC.fit(x_train, y_train)
print(accuracy_score(votingC.predict(x_test),y_test))
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)
results = pd.concat([test_PassengerId, test_survived],axis = 1)
results.to_csv("titanic.csv", index = False)