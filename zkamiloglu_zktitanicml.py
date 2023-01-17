# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



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
train=pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv("/kaggle/input/titanic/test.csv")

test_passengerId=test['PassengerId']
train.columns
train.describe()

 
train.info()

test.info()
def bar_plot(variable):

    var = train[variable]

    varValue = var.value_counts()

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index,varValue.index.values)

    plt.ylabel("number of sample")

    plt.title(variable)

    plt.show()

    print("{}:\n {}".format(variable, varValue))
Category1=['Survived','Pclass','Sex','SibSp','Parch','Embarked']

for c in Category1:

     bar_plot(c)  
Category2=['Cabin','Name','Ticket']

for c in Category2:

    print("{}:\n".format(train[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(train[variable] ,bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Number of sample")

    plt.title("{} distribution with hist".format(variable))

numericalVal=['Age','Fare']

for n in numericalVal:

    plot_hist(n)
#Pclass-Survived

train[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by='Survived',ascending=False)
#Sex-Survived

train[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by='Survived',ascending=False)
#SibSp-Survived

train[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by='Survived',ascending=False)
#Parch-Survived

train[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by='Survived',ascending=False)
#Embarked-Survived

train[["Embarked","Survived"]].groupby(["Embarked"],as_index=False).mean().sort_values(by='Survived',ascending=False)
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

    multiple_outliers = (i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers
train.loc[detect_outliers(train,["Age","SibSp","Parch","Fare"])]
train=train.drop(detect_outliers(train,['Age','SibSp','Parch','Fare']),axis=0).reset_index(drop=True)
train_len=len(train)
#combine my train and test at first



train=pd.concat([train,test],axis=0).reset_index(drop=True)
train.columns[train.isnull().any()]
#where isnull

train.isnull().sum()
#Isnull in Embarked

train[train["Embarked"].isnull()]
#reviewed the Fare features for Embarked

train.boxplot(column='Fare',by='Embarked')
#If I fill the spaces in the embarked featured according to the fare features

train['Embarked'] =train['Embarked'].fillna('C')

train[train["Embarked"].isnull()]
#look at Fare feature

train[train['Fare'].isnull()]

train[(train['Pclass'] == 3) & (train['Embarked'] == 'S')]['Fare'].mean()

# writing the missing data on the Fare

train['Fare'] =train['Fare'].fillna(13.64)  
list1=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

sns.heatmap(train[list1].corr(),annot=True, fmt=".2f")

plt.show()  
g=sns.factorplot(x='SibSp', y='Survived',data=train,kind='bar',size=6)

g.set_ylabels('Survived Probability')

plt.show()
g=sns.factorplot(x='Parch', y='Survived',data=train,kind='bar',size=6)

g.set_ylabels('Survived Probability')

plt.show()
g=sns.factorplot(x='Pclass', y='Survived',data=train,kind='bar',size=6)

g.set_ylabels('Survived Probability')

plt.show()
g=sns.FacetGrid(train,col='Survived')

g.map(sns.distplot,'Age',bins=25)

plt.show()
g=sns.FacetGrid(train,col='Survived')

g.map(sns.distplot,'Age',bins=25)

plt.xlim(15,35)

plt.show()



g=sns.FacetGrid(train,col='Survived')

g.map(sns.distplot,'Age',bins=25)

plt.ylim(0.00,0.05)

plt.show()



g=sns.FacetGrid(train,col='Survived',row='Pclass',size = 5)

g.map(plt.hist,'Age',bins=25)

g.add_legend()

plt.show()
g=sns.FacetGrid(train,row='Embarked',size=4)

g.map(sns.pointplot,'Pclass','Survived','Sex')

g.add_legend()

plt.show()
g=sns.FacetGrid(train,row='Embarked',col='Survived',size=4)

g.map(sns.barplot,'Sex','Fare')

g.add_legend()

plt.show()
train[train["Age"].isnull()]
kont=train[train['Age'].isnull()]

sns.factorplot(x='Sex', y='Age',data=train,kind='box')

plt.show()
sns.factorplot(x='Parch', y='Age',data=train,kind='box')

plt.show()



sns.factorplot(x='SibSp', y='Age',data=train,kind='box')

sns.factorplot(x = "Parch", y = "Age",data = train, kind = "box")

plt.show()


sns.heatmap(train[['Age','SibSp','Parch','Pclass']].corr(),annot = True)
index_nan_age=list(train['Age'][train['Age'].isnull()].index)

for i in index_nan_age:

    age_pred=train['Age'][((train['SibSp']==train.iloc[i]['SibSp'])&

                          (train['Parch']==train.iloc[i]['Parch'])&

                          (train['Pclass']==train.iloc[i]['Pclass']))].median()

    age_med=train['Age'].median()

    if not np.isnan(age_pred):

        train['Age'].iloc[i]=age_pred

    else:

        train['Age'].iloc[i]=age_med
train[train["Age"].isnull()]
train["Name"].head(10)
name = train["Name"]

train["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
train["Title"].head(10)
sns.countplot(x='Title',data=train)

plt.xticks(rotation = 60)

plt.show()
train["Title"] = train["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")

train["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle"  else 2 if i == "Mrs" else 3 if i == "Mr"else 4 for i in train["Title"]]

train["Title"].head()
sns.countplot(x='Title',data=train)

plt.xticks(rotation=60)

plt.show()
g=sns.factorplot(x='Title', y='Survived',data=train,kind='bar')

g.set_xticklabels(['Master',"Miss or Ms or Mlle",'Mrs',"Mr",'Other'])

g.set_ylabels('Survived Probability')

plt.show()
train=pd.get_dummies(train,columns=['Title'])

train.head()
train.drop(labels = ["Name"], axis = 1, inplace = True)
train.head()
train.head()
train['FamilySize']=train['Parch'] + train['SibSp'] + 1   
g=sns.factorplot(x='FamilySize', y='Survived',data=train,kind='bar')

g.set_ylabels('Survived Probabilty')

plt.show() 
train['FamilySize_Survived']=[1 if i<5 else 0 for i in train['FamilySize']]
g=sns.countplot(x='FamilySize_Survived',data=train)

g.set_xticklabels(['Big families','SmalFamilies'])

plt.show()
g=sns.factorplot(x='FamilySize_Survived', y='Survived',data=train,kind='bar')

g.set_xticklabels(['Big families','SmalFamilies'])

g.set_ylabels('Survival')

plt.show()
#FEATURE ENGİNEERİNG----> EMBARKED

train=pd.get_dummies(train,columns=['Embarked'])
#FEATURE ENGİNEERİNG----> TICKET

train.drop(labels=['Ticket'],axis=1,inplace=True)
#FEATURE ENGİNEERİNG----> PCLASS

train['Pclass']=train['Pclass'].astype('category')

train=pd.get_dummies(train,columns=['Pclass'])
#FEATURE ENGİNEERİNG----> SEX

train["Sex"]=train["Sex"].astype("category")

train=pd.get_dummies(train,columns=['Sex'])

train.head()
#FEATURE ENGİNEERİNG----> PASSENGERID,CABİN

train.drop(labels=['PassengerId','Cabin'],axis=1,inplace=True)
#FEATURE ENGİNEERİNG----> Family size

train = pd.get_dummies(train, columns= ["FamilySize_Survived"])

train.head()
train.columns
train.head()
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_len
test_data=train[train_len:]

test_data.drop(labels=['Survived'],axis=1,inplace=True)
test_data.head()
train_data = train[:train_len]

X_train = train_data.drop(labels="Survived", axis=1)

y_train=train_data["Survived"]

X_train, X_test,y_train, y_test = train_test_split(X_train,y_train,test_size=0.30,random_state = 42) 

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("train_data",len(train_data))
# call this function

logreg=LogisticRegression()

logreg.fit(X_train, y_train)

acc_log_train=round(logreg.score(X_train, y_train)*100,2)

acc_log_test=round(logreg.score(X_test, y_test)*100,2)

print("Training Accuarcy: %{}".format(acc_log_train))

print("Testing Accuarcy: %{}".format(acc_log_test))
random_state = 42

classifier = [DecisionTreeClassifier(random_state =  random_state),

             SVC(random_state = random_state),

             RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

            KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),

                 "max_depth" : range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],

                 "gamma":[0.001, 0.01, 0.1, 1],

                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features":[1,3,10],

                "min_samples_split": [2,3,10],

                "min_samples_leaf": [1,3,10],

                "n_estimators":[100,300],

                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors":np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform", "distance"],

                 "metric": ["euclidean","manhattan"]}

classifier_param = [dt_param_grid, svc_param_grid, rf_param_grid,logreg_param_grid,knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid = classifier_param[i], cv = StratifiedKFold(n_splits = 5), scoring = "accuracy",n_jobs= -1, verbose =1)

    clf.fit(X_train, y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
voting_classifier = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                                   ("rfc", best_estimators[2]),

                                                   ("lr",best_estimators[3])],

                                    voting="soft", n_jobs = -1)

voting_classifier =voting_classifier.fit(X_train,y_train)

print(accuracy_score(voting_classifier.predict(X_test),y_test))

#("c_svc",best_estimators[1]),
test_survived = pd.Series(voting_classifier.predict(test_data),name = "Survived").astype(int)

results = pd.concat([test_passengerId,test_survived],axis = 1)

results.to_csv("titanic.csv", index =False)
test_survived