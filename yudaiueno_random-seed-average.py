import pandas as pd

import numpy as np

import lightgbm as lgb

import warnings

from math import *

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from tqdm import tqdm



pd.set_option('display.max_columns', 200)

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train["is_test"] = False

test["is_test"] = True



data = pd.concat([test, train],axis = 0).loc[:,["PassengerId", "Survived", "is_test", 

                                                "Age", "Cabin", "Embarked", 

                                                "Fare", "Name", "Parch", "Pclass", 

                                                "Sex", "SibSp", "Ticket"]]



t = train.groupby(by = "Ticket", as_index=False).agg({"PassengerId" : 'count', "Sex" : lambda x : x[x == "female"].count()})

t.columns = ["Ticket", "SameTicket", "FemalesOnTicket"]

data = pd.merge(data, t, how = "left", on = "Ticket")



data["TicketDigits"] = pd.to_numeric(data["Ticket"].str.split(" ").str[-1], errors = "coerce").astype(np.str).str.len()

data["TicketIsNumber"] = ~data["Ticket"].str.contains("[A-Za-z]", regex = True)

data["FemalesPerTicketPart"] = data["FemalesOnTicket"] / data["SameTicket"]



data["DoubleName"] = data["Name"].str.contains("\(")

data["NameLen"] = data["Name"].str.len()



data["Title"] = data["Name"].str.split(", ").str[1].str.split(" ").str[0]

data.loc[data["Title"].str[-1] != ".", "Title"] = "Bad"

rare_title = data["Title"].value_counts()[data["Title"].value_counts() < 5].index

data["Title"] = data["Title"].apply(lambda x: 'Rare' if x in rare_title else x)



titletarget = data.groupby(by = "Title", as_index = False).agg({"Survived" : 'mean'})

titletarget.columns = ["Title", "TargetByTitle"]

data = pd.merge(data, titletarget, how = "left", on = "Title")



data["Family"] = data["Parch"] + data["SibSp"] + 1

data["Alone"] = data["Family"] == 1



cap = train.groupby(by = "Cabin", as_index = False).agg({"PassengerId" : 'count'})

cap.columns = ["Cabin", "SameCabin"]

data = pd.merge(data, cap, how = "left", on = "Cabin")

data["CabinNumber"] = pd.to_numeric(data["Cabin"].str[1:], errors = "coerce")

data["CabinEven"] = data["CabinNumber"] % 2

data["CabinsPerMan"] = data["Cabin"].str.split(" ").str.len()

data["Deck"] = data["Cabin"].str[0].rank().fillna(-1)



decktarget = data.groupby(by = "Deck", as_index = False).agg({"Survived" : 'mean'})

decktarget.columns = ["Deck", "TargetByDeck"]

data = pd.merge(data, decktarget, how = "left", on="Deck")



data['AgeGroup'] = pd.qcut(data['Age'].fillna(data['Age'].mean()).astype(int), 6)

agetarget = data.groupby(by = "AgeGroup", as_index = False).agg({"Survived" : 'mean'})

agetarget.columns = ["AgeGroup", "TargetByAgeGroup"]

data = pd.merge(data, agetarget, how = "left", on="AgeGroup")



data["IsTinyChild"] = data["Age"] < 1

data["IsChild"] = data["Age"] < 10

data["AverageAge"] = data["Age"] / data["Family"]



data['FareGroup'] = pd.qcut(data['Fare'].fillna(data['Fare'].mean()).astype(int), 6)

faretarget = data.groupby(by = "FareGroup", as_index = False).agg({"Survived" : 'mean'})

faretarget.columns = ["FareGroup", "TargetByFareGroup"]

data = pd.merge(data, faretarget, how = "left", on = "FareGroup")

data["AverageFareByFamily"] = data["Fare"] / data["Family"]

data["AverageFareByTicket"] = data["Fare"] / data["SameTicket"]

data["FareLog"] = np.log(data["Fare"])



pclasstarget = data.groupby(by = "Pclass", as_index = False).agg({"Survived" : 'mean'})

pclasstarget.columns = ["Pclass", "TargetByPclass"]

data = pd.merge(data, pclasstarget, how = "left", on = "Pclass")



Embarkedtarget = data.groupby(by = "Embarked", as_index = False).agg({"Survived" : 'mean'})

Embarkedtarget.columns = ["Embarked", "TargetByEmbarked"]

data = pd.merge(data, Embarkedtarget, how = "left", on = "Embarked")



Sextarget = data.groupby(by = "Sex", as_index = False).agg({"Survived" : 'mean'})

Sextarget.columns = ["Sex", "TargetBySex"]

data = pd.merge(data, Sextarget, how = "left", on = "Sex")



allfeatures = [

    "PassengerId", 

    "is_test", 

    "Survived", 

    "Age", 

    "Fare", 

    "Parch", 

    "Pclass",

    "SibSp", 

    "Sex", 

    "Embarked", 

    "SameTicket", 

    "FemalesOnTicket", 

    "SameCabin", 

    "Deck", 

    "TargetByDeck", 

    "TargetByTitle", 

    "TargetByAgeGroup", 

    "TargetByFareGroup",

    "TargetByPclass",

    "TargetByEmbarked",

    "TargetBySex",

    "Title", 

    "CabinNumber", 

    "CabinEven", 

    "CabinsPerMan", 

    "DoubleName", 

    "NameLen", 

    "TicketDigits",

    "TicketIsNumber",

    "IsTinyChild", 

    "IsChild", 

    "Alone", 

    "Family", 

    "AverageAge",

    "AverageFareByFamily",

    "AverageFareByTicket",

    "FemalesPerTicketPart"

]



data = data.loc[:, allfeatures]

data = pd.get_dummies(data, columns=[

    "Title", 

    "Embarked", 

    "Pclass",

    "Sex"

])



data.iloc[:,3:] = SimpleImputer(strategy = "most_frequent").fit_transform(data.iloc[:,3:])



data['DoubleName'] = data['DoubleName'].astype(float)

data['TicketIsNumber'] = data['TicketIsNumber'].astype(float)

data['IsTinyChild'] = data['IsTinyChild'].astype(float)

data['IsChild'] = data['IsChild'].astype(float)

data['Alone'] = data['Alone'].astype(float)



train_y = data[data["is_test"] == False].loc[:, ["Survived"]]

train_x = data[data["is_test"] == False].iloc[:, 3:]

X_test = data[data["is_test"] == True].iloc[:, 3:]

test_index = data[data["is_test"] == True].loc[:, "PassengerId"]



train_x.iloc[:,3:] = StandardScaler().fit_transform(train_x.iloc[:,3:])

X_test.iloc[:,3:] = StandardScaler().fit_transform(X_test.iloc[:,3:])
models = []

scores = []



for i in tqdm(range(0,90)):

    X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size = 0.40, random_state = i)

    

    model = lgb.LGBMClassifier(max_depth = 7,

                             lambda_l1 = 0.1,

                             lambda_l2 = 0.01,

                             learning_rate =  0.01, 

                             num_iterations=20000,

                             n_estimators = 5000, 

                             reg_alpha = 1.1, 

                             colsample_bytree = 0.9, 

                             subsample = 0.9,

                             n_jobs = 5, 

                             boosting='dart')

    

    model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], eval_metric = 'accuracy', verbose = False, early_stopping_rounds = 50);

    models.append(model)

    scores.append(accuracy_score(y_valid, model.predict(X_valid)))



lgb_mean = np.mean(scores)

lgb_std = np.std(scores)

print('正答率の平均：', lgb_mean)

print('正答率の標準偏差', lgb_std)
data = pd.DataFrame()

for i in range(0,90):

    data[i] = models[i].predict(X_test)



pred_test = np.round(data.mean(axis = 1))
sub = pd.DataFrame(test_index.astype(np.int), columns = ["PassengerId"])

sub["Survived"] = pred_test.astype(np.int)

sub.to_csv("sub_0527.csv", index = None)