import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

%matplotlib inline



train_df = pd.read_csv('../input/train.csv', header=0)        # Load the train file into a dataframe

test_df = pd.read_csv('../input/test.csv', header=0)        # Load the test file into a dataframe

#print('type(train_df)'+type(train_df)._name_)

#df_list = [train_df,test_df]



full_df = pd.concat([train_df,test_df], ignore_index=True)
train_df.head()
train_df.info()
test_df.head()
test_df.info()
train_df = pd.DataFrame()

test_df = pd.DataFrame()
full_df.info()
# get train_df, and test_df from full_df

def extract_df():

    tr_df= full_df.loc[full_df['Survived'].notnull()]

    tr_df.info()

    te_df = full_df.loc[full_df['Survived'].isnull()]

    te_df.info()

    return tr_df, te_df
train_df, test_df = extract_df()
# set(list(train_df)).symmetric_difference(list(test_df))
title_sr = full_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False) 

full_df['Title'] = title_sr

pd.crosstab(full_df['Title'], full_df['Sex'])
print(title_sr.value_counts())
title_list = set(title_sr)

print(title_list)

map_title_dic={"Mlle":"Miss", "Ms":"Miss", "Mme":"Mrs"}



working_dic = {}

for key in ['Lady', 'Countess','Capt', 'Col','Don', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:

    working_dic[key] = "Rare"

    

map_title_dic.update(working_dic)



full_df["Title"] = full_df["Title"].replace(map_title_dic)



print(set(list(full_df["Title"])))

SubCol01 = test_df.PassengerId # for submission

try: full_df.drop(["PassengerId","Name","Ticket","Cabin"], axis=1,inplace=True)  

except: print("except")

train_df, test_df = extract_df()
print (train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
feature_list = list(full_df)

print(feature_list)



for feature in feature_list:

    #print("------------")

    #print(feature)

    print(feature +"  "+ str(len(full_df[feature].value_counts())))
#find null sample

#train_df[train_df.isnull().any(axis=1)]
train_df.hist(bins="auto",figsize=(9,7),grid=False);
train_df.isnull().sum()
test_df.isnull().sum()
full_df["Embarked"].value_counts()
full_df["Embarked"].fillna("S", inplace=True)
full_df["Fare"].median()

full_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
train_df, test_df = extract_df()
full_df['Sex'] = full_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
def onehot(df,feature_list):

    print(df.shape)

    try:

        df = pd.get_dummies(df, columns=feature_list)

        print(df.shape)

        return df

    except:

        print("except")

onehot_list = ["Title","Pclass","Embarked"]

full_df = onehot(full_df,onehot_list)
full_df
train_df, test_df = extract_df()
train_df.head()
test_df.head()
X_train_age = full_df[[x for x in list(train_df) if not x in ["Survived"]]] # select use features



# split data for train

X_predict_age = X_train_age.loc[ X_train_age["Age"].isnull()]

X_train_age = X_train_age.loc[ X_train_age["Age"].notnull()] # use rows which age is not null

y_train_age = X_train_age.Age

try:

    X_train_age.drop("Age",axis=1,inplace = True)

    X_predict_age.drop("Age",axis=1,inplace = True)

except:

    print("except")

X_predict_age.head()
X_train_age.head()
from sklearn import preprocessing

scaler2 = preprocessing.StandardScaler().fit(X_train_age)

X_train_age = scaler2.transform(X_train_age)

X_predict_age = scaler2.transform(X_predict_age)

Age_None_list = full_df[full_df['Age'].isnull()].index.tolist()
X_train_age[1]
from sklearn.neural_network import MLPRegressor

mlr = MLPRegressor(solver='lbfgs', alpha=1e-5, 

                     hidden_layer_sizes=(50,50 ), random_state=1)

mlr.fit(X_train_age, y_train_age)          
mlr.score(X_train_age, y_train_age)    
plt.figure(figsize=(20,10))

plt.rc('font',size = 16)

plt.scatter(mlr.predict(X_train_age), y_train_age)

plt.xlabel('Smarts')

plt.ylabel('Probability')

plt.show()
for a, b in zip(np.array(y_train_age),mlr.predict(X_train_age)):

    print (a," ", b)
mlr.predict(X_predict_age)
#full_df["Age"][[1,2,5]]=[37,25,None]

full_df["Age"][Age_None_list] = mlr.predict(X_predict_age).tolist()

train_df, test_df = extract_df()
full_df
X_train = full_df[full_df["Survived"].notnull()]

X_train.head()
y_train = full_df["Survived"][full_df["Survived"].notnull()]

y_train.head()
X_predict = full_df[full_df["Survived"].isnull()]

X_predict.head()
try:

    X_train.drop("Survived",axis=1,inplace = True)

    X_predict.drop("Survived",axis=1,inplace = True)

except:

    print("except")







from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_predict = scaler.transform(X_predict)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,

                     hidden_layer_sizes=(50,50), random_state=1)

clf.fit(X_train, y_train)          
clf.score(X_train, y_train)    
clf.predict(X_train)
SubCol02 = clf.predict(X_predict).astype(int)

SubCol02
submission = pd.DataFrame({

        "PassengerId": SubCol01,

        "Survived": SubCol02

    })



submission.to_csv("titanic_submission.csv", index=False)
submission.head()