import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import fbeta_score

from sklearn.metrics import cohen_kappa_score

from sklearn.ensemble import BaggingClassifier



import warnings

warnings.filterwarnings('ignore')
# Read the data:

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

data = pd.concat([train,test],ignore_index=True)

labels=train["Survived"]
# look at data

#train.describe()

train.head(4)
test.head(4)
# Functions used in the kernel



# Create a graph that groups, counts and check survivors per group

def survival_rate(column,t):

    df=pd.DataFrame()

    df['total']=train.groupby(column).size()

    df['survived'] = train.groupby(column).sum()['Survived']

    df['percentage'] = round(df['survived']/df['total']*100,2)

    print(df)



    df['survived'].plot(kind=t)

    df['total'].plot(kind=t,alpha=0.5,title="Survivors per "+str(column))

    plt.show()



# If age is less than 1, we return 1. Else, we return the original age.

def normalize_age_below_one(age):

    if age < 1:

        return 1

    else:

        return age



# Group ages in buckets

def group_age(value):

    if value <= 10:

        return "0-10"

    elif value <= 20:

        return "10-20"

    elif value <= 30:

        return "20-30"

    elif value <= 40:

        return "30-40"

    elif value <= 50:

        return "40-50"

    elif value <= 60:

        return "50-60"

    elif value <= 70:

        return "60-70"

    elif value <= 80:

        return "70-80"

    elif value <= 90:

        return "80-90"

    else:

        return "No data"



# Change sex type to integers

def sex(value):

    if value == "male":

        return 0

    else:

        return 1



# Change embarked type to integers

def embarked(value):

    if value == "C":

        return 0

    elif value =="Q":

        return 1

    else:

        return 2



# Clean title and convert to numeric.

data["TitleClean"] = data["Name"].str.extract('(\w*\.)', expand=True)

def title_to_int(value):

    if value == "Capt.":

        return 0

    elif value == "Col.":

        return 1

    elif value == "Countess.":

        return 2

    elif value == "Don.":

        return 3

    elif value == "Dr.":

        return 4

    elif value == "Jonkheer.":

        return 5

    elif value == "Lady.":

        return 6

    elif value == "Major.":

        return 7

    elif value == "Master.":

        return 8

    elif value == "Miss.":

        return 9

    elif value == "Mlle.": #Same as miss

        return 9

    elif value == "Mme.":

        return 11

    elif value == "Mr.":

        return 12

    elif value == "Mrs.":

        return 13

    elif value == "Ms.":

        return 14

    elif value == "Rev.":

        return 15

    elif value == "Sir.":

        return 16

    elif value == "Dona.": # Same as Mrs

        return 13

    else:

        return np.nan

    

# Test a bunch of models. If NL is false, Neural Networks are not tested (they are pretty slow)

def lets_try(NL):

    results={}

    def test_model(clf):

        

        cv = KFold(n_splits=10)

        fbeta_scorer = make_scorer(fbeta_score, beta=1)

        cohen_scorer = make_scorer(cohen_kappa_score)

        accu = cross_val_score(clf, features, labels, cv=cv)

        fbeta = cross_val_score(clf, features, labels, cv=cv,scoring=fbeta_scorer)

        cohen = cross_val_score(clf, features, labels, cv=cv,scoring=cohen_scorer)

        scores=[accu.mean(),fbeta.mean(),cohen.mean()]

        return scores



    # Decision Tree

    clf = tree.DecisionTreeClassifier()

    results["Decision Tree"]=test_model(clf)

    # Logistic Regression

    clf = LogisticRegression()

    results["Logistic Regression"]=test_model(clf)

    # SVM Linear

    clf = svm.LinearSVC()

    results["Linear SVM"]=test_model(clf)

    # SVM RBF

    clf = svm.SVC()

    results["RBF SVM"]=test_model(clf)

    # Gaussian Bayes

    clf = GaussianNB()

    results["Gaussian Naive Bayes"]=test_model(clf)

    # Random Forest

    clf=RandomForestClassifier()

    results["Random Forest"]=test_model(clf)

    # AdaBoost with Decision Trees

    clf=AdaBoostClassifier()

    results["AdaBoost"]=test_model(clf)

    # SGDC

    clf=SGDClassifier()

    results["SGDC"]=test_model(clf)

    # Bagging

    clf=BaggingClassifier()

    results["Bagging"]=test_model(clf)

    # Neural Networks

    if NL:

        clf=MLPClassifier()

        results["Neural Network"]=test_model(clf)

    

    results = pd.DataFrame.from_dict(results,orient='index')

    results.columns=["Accuracy","F-Score", "Cohen Kappa"] 

    results=results.sort(columns=["Accuracy","F-Score", "Cohen Kappa"],ascending=False)

    results.plot(kind="bar",title="Model Scores")

    axes = plt.gca()

    axes.set_ylim([0,1])

    return plt
# Count the number of rows

print("*** Number of rows: " + str(train.shape[0]))

total = train.shape[0]

print("\n")



# List all the columns

print("*** Columns: " + str(train.columns.values), end="\n")



# Count the number of NaNs each column has.

print("\n*** NaNs per column:")

print(pd.isnull(train).sum())
# Change gender's text to integers

data["Sex"] = data["Sex"].apply(sex)



# Draw survival per sex

survival_rate("Sex","barh")
# Draw survival per Class

survival_rate("Pclass","barh")
print("*** Number of people with age less than 1 (months):")

print(train[train["Age"] < 1.0].shape[0])



# Those with age <1, changed to 1

data['Age'] = data['Age'].apply(normalize_age_below_one)



# Create new feature with data in buckets

data["AgeGroup"] = data["Age"].apply(group_age)

train["AgeGroup"] = train["Age"].apply(group_age)



# Draw survival per age group

survival_rate("AgeGroup","bar")
# Get Fare statistics

print("*** Fare statistics:")

print(train["Fare"].describe())



# Seems that some people paid nothing:

print("\n*** People with fare 0:")

nothing = train[train["Fare"] == 0]

print(nothing[["Name","Sex","Age","Pclass","Survived"]])



# Graph average Fare per Class

train.groupby("Pclass").mean()['Fare'].plot(kind="bar",title="Average Fare per Class")

plt.show()
# Change embarkation data type to integers

data["Embarked"] = data["Embarked"].apply(embarked)



# Graph survived per port of embarkation

survival_rate("Embarked","bar")
data["FamilyMembers"]=data["SibSp"]+data["Parch"]

train["FamilyMembers"]=train["SibSp"]+data["Parch"]



print("*** Family statistics, members:")

print("Min: " + str(train["FamilyMembers"].min()))

print("Average: " + str(round(train["FamilyMembers"].mean(),2)))

print("Max: " + str(train["FamilyMembers"].max()), end="\n\n")



print("*** Average family members per Class:")

print(train.groupby("Pclass").mean()['FamilyMembers'], end="\n\n")



# Families with more than 5 members

large_families=train[train["FamilyMembers"]>= 5]

large_families_by_ticket=large_families.groupby("Ticket").sum()['Survived']

print("*** Large families by ticket. Did all family die?:")

print(large_families_by_ticket==0, end="\n\n")



# Largest family where all members died

largest_family_ticket=train["Ticket"][train["FamilyMembers"]==10].iloc[0]

name=train["Name"][train["Ticket"]==largest_family_ticket].iloc[0]

print("*** Largest family, all members died: "+ name.split(",")[0], end="\n\n")

# More info: http://www.bbc.com/news/uk-england-cambridgeshire-17596264



survival_rate("FamilyMembers","bar")
train["Ticket"].head()
data["TicketClean"] = data["Ticket"].str.extract('(\d{2,})', expand=True)

data["TicketClean"].head()
print("Rows with NaN: " + str(pd.isnull(data["TicketClean"]).nonzero()[0]))

print("Ticket number: ")

print(str(data["Ticket"].ix[179]))

print(str(data["Ticket"].ix[271]))

print(str(data["Ticket"].ix[302]))

print(str(data["Ticket"].ix[597]))

print(str(data["Ticket"].ix[772]))

print(str(data["Ticket"].ix[841]))

print(str(data["Ticket"].ix[1077]))

print(str(data["Ticket"].ix[1193]))
data["TicketClean"] = data["Ticket"].str.extract('(\d{3,})', expand=True)

data["TicketClean"] = data["TicketClean"].apply(pd.to_numeric)

med1=data["TicketClean"].median()

med2=data["TicketClean"].median()+data["TicketClean"].std()

med3=data["TicketClean"].median()-data["TicketClean"].std()

data.set_value(179, 'TicketClean', int(med1))

data.set_value(271, 'TicketClean', int(med1))

data.set_value(302, 'TicketClean', int(med1))

data.set_value(597, 'TicketClean', int(med1))

data.set_value(772, 'TicketClean', int(med2))

data.set_value(841, 'TicketClean', int(med2))

data.set_value(1077, 'TicketClean', int(med2))

data.set_value(1193, 'TicketClean', int(med2))

data["TicketClean"].head()
data["TitleClean"] = data["Name"].str.extract('(\w*\.)', expand=True)

data.groupby(data["TitleClean"]).size()
data["TitleClean"] = data["TitleClean"].apply(title_to_int)
df=pd.DataFrame()

df['total']=train.groupby("Survived").size()

df=df['total']/train.shape[0]

df.plot(kind="bar",title="Label's Balance")

axes = plt.gca()

axes.set_ylim([0,1])

plt.show()
data.head()
remove=['Name','Cabin','Ticket', 'AgeGroup']

for column in remove:

    data = data.drop(column, 1)



# Add missing ages. If there is a NaN, change it with the average for that title group.

list_nan=pd.isnull(data["Age"]).nonzero()

# Get a pd with the mean age for each title

means = data.groupby("TitleClean").mean()['Age']

# for each row with NaN, we write the average

for i in list_nan[0]:

    temp_title = data["TitleClean"].ix[i]

    data.set_value(i, 'Age', int(means[temp_title]))



# Add missing fare. If there is a NaN, change it with the average for that Pclass.

list_nan=pd.isnull(data["Fare"]).nonzero()

# Get a pd with the mean age for each title

means = data.groupby("Pclass").mean()['Fare']

# for each row with NaN, we write the average

for i in list_nan[0]:

    temp_class = data["Pclass"].ix[i]

    data.set_value(i, 'Fare', int(means[temp_class]))
# Prepare features

train=data[data['Survived'].isin([0, 1])]

#labels=train["Survived"]

train=train.drop("Survived", 1)

train=train.drop('PassengerId', 1)

features=train



# Prepare testing data

test=data[~data['Survived'].isin([0, 1])]

test=test.drop("Survived", 1)
lets_try(NL=False).show()
def draw_best_features():

    clf=RandomForestClassifier()

    clf.fit(features,labels)

    importances = clf.feature_importances_

    names=features.columns.values



    pd.Series(importances*100, index=names).plot(kind="bar")

    plt.show()

    

draw_best_features()
# Now let's test only with relevant features

#best_features=["Pclass","Sex","Age","Fare","FamilyMembers", "TicketClean", "TitleClean"]

best_features=["Pclass","Sex","Age","Fare", "TicketClean", "TitleClean"]

features=features[best_features]

features.head()
scaler = MinMaxScaler()

features_backup=features

features = scaler.fit_transform(features)

pd.DataFrame(features).head()
lets_try(NL=False).show()
features=features_backup

cv = KFold(n_splits=5)



parameters = {'n_estimators': [10,20,30,40,50],

               'min_samples_split' :[2,3,4,5],

               'min_samples_leaf' : [1,2,3]

             }



clf = RandomForestClassifier()

grid_obj = GridSearchCV(clf, parameters, cv=cv)

grid_fit = grid_obj.fit(features, labels)

best_clf = grid_fit.best_estimator_ 



best_clf.fit(features,labels)
PassengerId=test["PassengerId"]
#remove=['PassengerId','SibSp', 'Parch', 'Embarked']

#for column in remove:

#    test = test.drop(column, 1)
test=test[best_features]

test.head()
predictions=best_clf.predict(test)



sub = pd.DataFrame({

        "PassengerId": PassengerId,

        "Survived": predictions

    })

sub.to_csv("titanic_submission.csv", index=False)