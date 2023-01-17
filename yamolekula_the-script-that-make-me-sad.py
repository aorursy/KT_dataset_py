%matplotlib inline
import pandas as pd
import scipy.stats as stats
from pandas.core.frame import DataFrame
import re
import operator
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold,StratifiedKFold
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import time
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from time import time
from operator import itemgetter
from sklearn.ensemble import BaggingClassifier

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/train.csv") 
test = pd.read_csv("../input/test.csv")
pid = "PassengerId"
feature_name = ["Pclass",# Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
                "SibSp",# Siblings & Spouses
                "Parch",# Parents/Children
                "Sex"]
# the features wich we'll add in the first turn:
# "Fare"  - because it has only one NA-value
# "Embarked" - because it has only 2 NA-values
# "FamilySize" - because it a simple feature engineering

# the features wich we'll add in the second turn:
# "Age" - it has many NA

# the features that we won't add, but we'll extract data from them
# "Cabin"
# "Ticket"
# "Name"

# artificial features - the result of feature engineering:
# "Title"      # part of feature Name: Mr, Miss,Master etc
# "FamilyId"   # Surname plus FamilySize
# "Mother"
# "Child"
# "Deck"
# "CabinNum"
# "CabinPos"

target = "Survived"
test[target] = None
test["Cat"] = "test"
train["Cat"] = "train"
full = pd.concat([train, test], axis=0, ignore_index=True)
full.describe()
print("The percent of filled data (%):")
full.count()/full.shape[0]*100
full.shape[0]-full.count()
# пол
full.loc[full["Sex"] == "male", "Sex"] = 1
full.loc[full["Sex"] == "female", "Sex"] = 0
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

rf = RandomForestRegressor()
fare_col = ["Pclass", "Title", "Sex", "SibSp", "Parch"]
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
feature_name.append("Fare")
full["Embarked"] = full["Embarked"].fillna("S")
full.loc[full["Embarked"] == "S", "Embarked"] = 0
full.loc[full["Embarked"] == "C", "Embarked"] = 1
full.loc[full["Embarked"] == "Q", "Embarked"] = 2
feature_name.append("Embarked")
full["FamilySize"] = full["SibSp"] + full["Parch"]
feature_name.append("FamilySize")
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

titles = full["Name"].apply(get_title)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3,
                 "Master": 4, "Dr": 5, "Rev": 6,
                 "Major": 7, "Col": 7, "Mlle": 8,
                 "Mme": 8, "Don": 9, "Lady": 10,
                 "Countess": 10, "Jonkheer": 10,
                 "Sir": 9, "Capt": 7, "Ms": 2,
                 "Dona": 10}
for k, v in title_mapping.items():
    titles[titles == k] = v
full["Title"] = titles
feature_name.append("Title")
full.Age=full.groupby("Title").transform(lambda x: x.fillna(x.median()))
feature_name.append("Age")
print("percent of filled data in feature Age: {}%".format(full.Age.count()/full.shape[0]*100))
family_id_mapping = {}

# group by families --> family_id_mapping
def get_family_id(row):
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name, row["FamilySize"]) 
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            current_id = (
                max(family_id_mapping.items(), 
                    key=operator.itemgetter(1) 
                    )[1] + 1
            )
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]  

family_ids = full.apply(get_family_id, axis=1)
family_ids[full["FamilySize"] < 3] = -1
full["FamilyId"] = family_ids
feature_name.append("FamilyId")
full["Mother"] = 0
moth_idx = (full["Sex"] == 0) & (full["Parch"] > 0) & (full["Age"] > 18) & (full["Title"] == 2)
full.loc[moth_idx, "Mother"] = 1
feature_name.append("Mother")

full["Child"] = 0
full.loc[(full.Parch > 0) & (full["Age"] <= 18), "Child"] = 1
feature_name.append("Child")
deck_mapping = {}

# full deck_mapping dict
def get_deck(name):
    if name not in deck_mapping:
        if len(deck_mapping) == 0:
            current_id = 1
        else:
            current_id = (
                max(deck_mapping.items(),  
                    key=operator.itemgetter(1)  
                    )[1] + 1  
            )
        deck_mapping[name] = current_id  
    return deck_mapping[name] 


def get_cabin_pos(cabin):
    cabin_pos = list(filter(None, re.split("[^0-9]", cabin)))
    if cabin_pos:
        return int(cabin_pos[0])
    else:
        return 0

full["StrDeck"] = "UNK"
full.loc[full["Cabin"].notnull(), "StrDeck"] = full.loc[full["Cabin"].notnull(), "Cabin"].apply(lambda x: x[0])
full["Deck"] = full["StrDeck"].apply(get_deck)
feature_name.append("Deck")

full["CabinNum"] = -1
full.loc[full["Cabin"].notnull(), "CabinNum"] = full.loc[full["Cabin"].notnull(), "Cabin"].apply(get_cabin_pos)
feature_name.append("CabinNum")

km = KMeans(n_clusters=3)

km.fit(full["CabinNum"].reshape(-1, 1))
full["CabinPos"] = km.predict(full["CabinNum"].reshape(-1, 1).astype(float))
feature_name.append("CabinPos")
def get_ticket_title(x:str):
    arr = x.split(" ")
    if len(arr)>1:
        return arr[0]
    else:
        return "UNK"
    
full["TicketTitle1"]=full.Ticket.apply(get_ticket_title)

full.loc[full.TicketTitle1.apply(lambda x:x in ["A./5.","A/5", "A/S", "A.5.", "A/5." ]),"TicketTitle1"]= "A5"
full.loc[full.TicketTitle1.apply(lambda x:x in ["C.A.","CA","CA.","C.A./SOTON"]),"TicketTitle1"]= "CA"
full.loc[full.TicketTitle1.apply(lambda x:x in ["A/4","A/4.","A4.","S.C./A.4.","SC/A4"]),"TicketTitle1"]= "A4"
full.loc[full.TicketTitle1.apply(lambda x:x in ["W./C.","W/C"]),"TicketTitle1"]= "WC"
full.loc[full.TicketTitle1.apply(lambda x:x in ["P/PP","PP","S.O./P.P.","S.W./PP","SW/PP"]),"TicketTitle1"]= "PP"
full.loc[full.TicketTitle1.apply(lambda x:x in ["W.E.P.","WE/P"]),"TicketTitle1"]= "WEP"
full.loc[full.TicketTitle1.apply(lambda x:x in ["F.C.","F.C.C."]),"TicketTitle1"]= "FC"
full.loc[full.TicketTitle1.apply(lambda x:x in ["S.O.C.","SO/C"]),"TicketTitle1"]= "SOC"
full.loc[full.TicketTitle1.apply(lambda x:x in ["SC","SC/AH","SC/Paris","SC/PARIS","S.C./A.4.","SC/A.3"]),"TicketTitle2"]= "SC"
full.loc[full.TicketTitle1.apply(lambda x:x in ["SC","SC/AH","SC/Paris","SC/PARIS","S.C./A.4.","SC/A.3"]),"TicketTitle1"]= "UNK"
full.loc[full.TicketTitle1.apply(lambda x:x in ["SOTON/O.Q.","SOTON/O2","STON/O","SOTON/OQ","STON/O2.","STON/OQ."]),"TicketTitle2"]= "SOTON"
full.loc[full.TicketTitle1.apply(lambda x:x in ["SOTON/O.Q.","SOTON/O2","STON/O","SOTON/OQ","STON/O2.","STON/OQ."]),"TicketTitle1"]= "UNK"
full.loc[full.TicketTitle1.apply(lambda x:x in ["S.C./PARIS","SC/PARIS","SC/Paris"]),"TicketTitle3"]= "PARIS"
full.loc[full.TicketTitle1.apply(lambda x:x in ["S.C./PARIS","SC/PARIS","SC/Paris"]),"TicketTitle2"]= "UNK"
full.loc[full.TicketTitle1.apply(lambda x:x in ["S.C./PARIS","SC/PARIS","SC/Paris"]),"TicketTitle1"]= "UNK"

full.TicketTitle1=full.TicketTitle1.fillna("UNK")
full.TicketTitle2=full.TicketTitle2.fillna("UNK")
full.TicketTitle3=full.TicketTitle3.fillna("UNK")

for name in full.TicketTitle1.unique():
    print("{} - {}".format(name,full.TicketTitle1[full.TicketTitle1==name].count()))
print("---------------------------------")
for name in full.TicketTitle2.unique():
    print("{} - {}".format(name,full.TicketTitle2[full.TicketTitle2==name].count()))
print("---------------------------------")
for name in full.TicketTitle3.unique():
    print("{} - {}".format(name,full.TicketTitle3[full.TicketTitle3==name].count()))

for name in full.TicketTitle1.unique():
    if full.TicketTitle1[full.TicketTitle1==name].count()<5:
        full.loc[full.TicketTitle1==name,"TicketTitle1"]="UNK"

for name in full.TicketTitle2.unique():
    if full.TicketTitle2[full.TicketTitle2==name].count()<5:
        full.loc[full.TicketTitle2==name,"TicketTitle2"]="UNK"

for name in full.TicketTitle3.unique():
    if full.TicketTitle3[full.TicketTitle3==name].count()<5:
        full.loc[full.TicketTitle3==name,"TicketTitle3"]="UNK"
        
feature_name.append("TicketTitle1")
feature_name.append("TicketTitle2")
feature_name.append("TicketTitle3")
