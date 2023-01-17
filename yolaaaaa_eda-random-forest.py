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
train = pd.read_csv("/kaggle/input/titanic/train.csv")

gender_sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.info()
for a in train["Name"]:

    print(a)
train["Embarked"].value_counts()
# Import LabelEncoder

from sklearn import preprocessing

#creating labelEncoder

le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
train2 = train.copy()

train2['Sex'] = train2['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

train2["Embarked"] = train2["Embarked"].fillna("S")

train2['Embarked'] = train2['Embarked'].map( {'S': 1, 'C': 2, 'Q': 3} ).astype(int)

train2["Fam member"] = train2["SibSp"] + train2["Parch"]
train2.head()
# overwriting column with replaced value of age  

train2["Name"]= train2["Name"].str.replace("Mlle", "Miss", case = False)

train2["Name"]= train2["Name"].str.replace("Ms", "Miss", case = False)

train2["Name"]= train2["Name"].str.replace("Mme", "Mrs", case = False)
train2["Mr"]=train2["Name"].str.contains(pat = "Mr\. ", regex=True ) 

train2["Mrs"]=train2["Name"].str.contains(pat = "Mrs\. ", regex=True) 

train2["Miss"]=train2["Name"].str.contains(pat = "Miss\. ", regex=True)

train2["Title"]=train2["Name"].str.contains(pat = "Col\. |Sir\. |Major|Countess|Capt|Don\. |Dr\. |Rev\. |Jonkheer\.|Dona\. |Lady\.", regex=True)

train2["Master"]=train2["Name"].str.contains(pat = "Master\. ", regex=True)
train2.loc[train2['Mr'] == True, 'all title'] = 1

train2.loc[train2['Mrs'] == True, 'all title'] = 2

train2.loc[train2['Miss'] == True, 'all title'] = 3

train2.loc[train2['Title'] == True, 'all title'] = 4

train2.loc[train2['Master'] == True, 'all title'] = 5
train3 = train2.copy()

train3.head()
train3.info()
# train3
train3["ticket contains string"] =train3["Ticket"].str.contains(pat = '\D+', regex=True)
train3.loc[train3['ticket contains string'] == True, 'ticket contains string'] = 1

train3.loc[train3['ticket contains string'] != True, 'ticket contains string'] = 0
train3.head()
import re
def get_code_ticket(ticket):

    found = re.findall('^[^0-9][^\s]+', ticket)

    if len(found) > 0 :

        found2 = found[0].replace(".","")

        return found2

    else:

        return "0"
get_code_ticket("SOTON/OQ 392086")
train3["ticket code"] = train3["Ticket"].apply(get_code_ticket)
train3.head(2)
code = le.fit_transform(train3["ticket code"]).tolist()

train3["tckt code"] = code
train3["ticket code"].value_counts()
train3.info()
train3[train3["Age"].isnull()]
train3["all title"].value_counts()
mean_age_mr = round(train3["Age"][train3["Mr"]].mean())

mean_age_mrs = round(train3["Age"][train3["Mrs"]].mean())

mean_age_miss = round(train3["Age"][train3["Miss"]].mean())

mean_age_master = round(train3["Age"][train3["Master"]].mean())

mean_age_title = round(train3["Age"][train3["Title"]].mean())
not_null_age = train3[train3["Age"].notnull()]

not_null_age
null_age_mr = train3[(train3["all title"]==True) & (train3["Age"].isnull())]

null_age_mr["Age"] = null_age_mr["Age"].fillna(mean_age_mr)

null_age_mr
null_age_mrs = train3[(train3["Mrs"]==True) & (train3["Age"].isnull())]

null_age_mrs["Age"] = null_age_mrs["Age"].fillna(mean_age_mrs)

null_age_mrs
null_age_miss = train3[(train3["Miss"]==True) & (train3["Age"].isnull())]

null_age_miss["Age"] = null_age_miss["Age"].fillna(mean_age_miss)

null_age_miss
null_age_master = train3[(train3["Master"]==True) & (train3["Age"].isnull())]

null_age_master["Age"] = null_age_master["Age"].fillna(mean_age_master)

null_age_master
null_age_title = train3[(train3["Title"]==True) & (train3["Age"].isnull())]

null_age_title["Age"] = null_age_title["Age"].fillna(mean_age_title)

null_age_title
train4 = pd.concat([not_null_age,null_age_master,null_age_miss,null_age_mr,null_age_mrs,null_age_title])

train4
train4.describe()
train3.columns
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
colormap = plt.cm.RdBu

plt.figure(figsize=(8,6))

plt.title('Correlation of Features', y=1.05, size=15)

sns.heatmap(train4.copy().drop(columns=["Name", "Ticket","Cabin","PassengerId","ticket code"]).astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
train4.columns
fitur2 = ['Pclass', 'Sex', 'Age', 'SibSp',

       'Parch','Fare', 'Embarked', 'Fam member', 'Mr',

       'Mrs', 'Miss', 'Title', 'Master', 'all title', 'ticket contains string',

        'tckt code']
fitur = ['Pclass', 'Sex', 'SibSp',

       'Parch', 'Fare', 'Embarked', 'all title', 'ticket contains string']
from sklearn.ensemble import RandomForestClassifier

RFC_METRIC = 'gini'  #metric used for RandomForrestClassifier

NUM_ESTIMATORS = 100 #number of estimators used for RandomForrestClassifier

NO_JOBS = 4 #number of parallel jobs used for RandomForrestClassifier





RANDOM_STATE = 42



clf = RandomForestClassifier(n_jobs=NO_JOBS, 

                             random_state=RANDOM_STATE,

                             criterion=RFC_METRIC,

                             n_estimators=NUM_ESTIMATORS,

                             verbose=False)
clf.fit(train4[fitur2],  train4["Survived"].values)
preds = clf.predict(train4[fitur2])
tmp = pd.DataFrame({'Feature': fitur2, 'Feature importance': clf.feature_importances_})

tmp = tmp.sort_values(by='Feature importance',ascending=False)

plt.figure(figsize = (7,4))

plt.title('Features importance',fontsize=14)

s = sns.barplot(x='Feature',y='Feature importance',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.show()
train4.info()
# Import the model we are using

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data

rf.fit(train4[fitur],  train4["Survived"].values.ravel())
test2 = train.copy()

test2['Sex'] = test2['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

test2["Embarked"] = test2["Embarked"].fillna("S")

test2['Embarked'] = test2['Embarked'].map( {'S': 1, 'C': 2, 'Q': 3} ).astype(int)



# overwriting column with replaced value of age  

test2["Name"]= test2["Name"].str.replace("Mlle", "Miss", case = False)

test2["Name"]= test2["Name"].str.replace("Ms", "Miss", case = False)

test2["Name"]= test2["Name"].str.replace("Mme", "Mrs", case = False)



test2["Mr"]=test2["Name"].str.contains(pat = 'Mr\. ') 

test2["Mrs"]=test2["Name"].str.contains(pat = 'Mrs\. ') 

test2["Miss"]=test2["Name"].str.contains(pat = 'Miss\. ')

test2["Title"]=test2["Name"].str.contains(pat = "Col\. |Sir\. |Major|Countess|Capt|Don\. |Dr\. |Rev\. |Jonkheer\. |Dona\. |Lady\. ", regex=True)

test2["Master"]=test2["Name"].str.contains(pat = 'Master\. ')



test2.loc[test2['Mr'] == True, 'all title'] = 1

test2.loc[test2['Mrs'] == True, 'all title'] = 2

test2.loc[test2['Miss'] == True, 'all title'] = 3

test2.loc[test2['Title'] == True, 'all title'] = 4

test2.loc[test2['Master'] == True, 'all title'] = 5



test2["Fam member"] = test2["SibSp"] + test2["Parch"]



test3 = test2.copy()

test3.head()
test3["ticket contains string"] =test3["Ticket"].str.contains(pat = '\D+', regex=True)



test3.loc[test3['ticket contains string'] == True, 'ticket contains string'] = 1

test3.loc[test3['ticket contains string'] != True, 'ticket contains string'] = 0

test3.head()
test3["ticket code"] = test3["Ticket"].apply(get_code_ticket)

code = le.fit_transform(test3["ticket code"]).tolist()

test3["tckt code"] = code
test3.info()
mean_age_mr_t = round(test3["Age"][test3["Mr"]].mean())

mean_age_mrs_t = round(test3["Age"][test3["Mrs"]].mean())

mean_age_miss_t = round(test3["Age"][test3["Miss"]].mean())

mean_age_master_t = round(test3["Age"][test3["Master"]].mean())

mean_age_title_t = round(test3["Age"][test3["Title"]].mean())





not_null_age_t = test3[test3["Age"].notnull()]

# not_null_age_t





null_age_mr_t = test3[(test3["all title"]==True) & (test3["Age"].isnull())]

null_age_mr_t["Age"] = null_age_mr_t["Age"].fillna(mean_age_mr_t)

# null_age_mr_t





null_age_mrs_t = test3[(test3["Mrs"]==True) & (test3["Age"].isnull())]

null_age_mrs_t["Age"] = null_age_mrs_t["Age"].fillna(mean_age_mrs_t)

# null_age_mrs_t





null_age_miss_t = test3[(test3["Miss"]==True) & (test3["Age"].isnull())]

null_age_miss_t["Age"] = null_age_miss_t["Age"].fillna(mean_age_miss_t)

# null_age_miss_t







null_age_master_t = test3[(test3["Master"]==True) & (test3["Age"].isnull())]

null_age_master_t["Age"] = null_age_master_t["Age"].fillna(mean_age_master_t)

# null_age_master_t





null_age_title_t = test3[(test3["Title"]==True) & (test3["Age"].isnull())]

null_age_title_t["Age"] = null_age_title_t["Age"].fillna(mean_age_title_t)

# null_age_title_t



test4 = pd.concat([not_null_age_t,null_age_master_t,null_age_miss_t,null_age_mr_t,null_age_mrs_t,null_age_title_t])

test4
test4.info()
test4["Fare"] = test4["Fare"].fillna(0)
test4.info()
test4
y_test_han = rf.predict(test4[fitur])



y_test_han = pd.DataFrame(data=y_test_han)

y_test_han = y_test_han.round()

y_test_han.astype(int)
hasil = pd.DataFrame()

hasil["PassengerId"] = test["PassengerId"]

hasil["Survived"] = y_test_han.astype(int)

hasil.to_csv("coba9.csv", index=False)