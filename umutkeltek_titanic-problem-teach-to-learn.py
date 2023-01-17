import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn





%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

train_test = pd.concat([train, test], ignore_index=True, sort  = False)
train_test.head()
print('Train data shape is: ', train.shape)

print('Test data shape is: ', test.shape)

print('Mixed data shape is: ', train_test.shape)
survivedd = train[train["Survived"] == 1]

not_survived = train[train["Survived"] == 0]



print("Survived count: {x} {y:1.3f} %".format(x=len(survivedd), y=float(len(survivedd) / len(train)) * 100))

print("Not Survived count: {z} {n:1.3f} %".format(z=len(not_survived), n=float(len(not_survived) / len(train) * 100)))
ax=sns.countplot(train["Survived"])



#Just fancy code to write percentages on bar.

totals = []

for i in ax.patches:

    totals.append(i.get_height())



total = sum(totals)

for i in ax.patches:

    # get_x pulls left or right; get_height pushes up or down

    ax.text(i.get_x()+.25, i.get_height()-250.95, \

            str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,

                color='white')



plt.title('Survival Status')



#Only %38.38 of passengers survived.
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

train_test['FamilySize'] = train_test['SibSp'] + train_test['Parch'] + 1



train['IsAlone'] = 0

train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1



train_test['IsAlone'] = 0

train_test.loc[train_test['FamilySize'] == 1, 'IsAlone'] = 1



test['IsAlone'] = 0

test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1



# We have two new feature.

train_test.corr()
plt.figure(figsize=(16,5))

corr_map = sns.heatmap(train_test.corr(),annot=True)
fig, ax = plt.subplots(figsize=(12,6),ncols=2,nrows=2)

ax1 = sns.barplot(x="SibSp",y="Survived", data=train_test, ax = ax[0][0]);

ax2 = sns.barplot(x="Parch",y="Survived", data=train_test, ax = ax[0][1]);

ax3 = sns.countplot(x="SibSp", data=train_test, ax = ax[1][0]);

ax4 = sns.countplot(x="Parch", data=train_test, ax = ax[1][1]);

#ax3.set_yscale('log')

#ax4.set_yscale('log')

ncount = len(train["SibSp"])

for p in ax3.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax3.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

            ha='center', va='bottom')

for p in ax4.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax4.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

            ha='center', va='bottom')
fig, ax = plt.subplots(figsize=(16,12),ncols=2,nrows=2)

ax1 = sns.barplot(x="Sex",y="Survived", data=train, ax = ax[0][0]);

ax2 = sns.barplot(x="Pclass",y="Survived", data=train, ax = ax[0][1]);

ax3 = sns.barplot(x="FamilySize",y="Survived", data=train, ax = ax[1][0]);

ax4 = sns.barplot(x="Embarked",y="Survived", data=train, ax = ax[1][1]);
print("Missing Data in Train Set"+'\n')

total = train.isnull().sum().sort_values(ascending = False)

percentage = total/len(train)*100

missing_data = pd.concat([total, percentage], axis=1, keys=['Total', '%'])

print(missing_data.head(3))



print('\n')

print("Missing Data in Test Set"+'\n')

total1 = test.isnull().sum().sort_values(ascending = False)

percentage1 = total1/len(test)*100

missing_data1 = pd.concat([total1, percentage1], axis=1, keys=['Total', '%'])

print(missing_data1.head(3))

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="magma")

#yticklabes=False removes left side labels.

#cbar=False removes the colorbar.
display(train_test[train_test.Embarked.isnull()])
embarked_missing = train_test[(train_test["Sex"] == "female") & (train_test["Pclass"] ==1) & (train_test["Cabin"].str.startswith('B')) 

                        & (train_test["Fare"]>70) & (train_test["Fare"]<100)]

# Embarked_missing shows us passengers who have similar values.

print(embarked_missing["Embarked"].value_counts())



train["Embarked"] = train["Embarked"].fillna("S")

train_test["Embarked"] = train_test["Embarked"].fillna("S")
fare_missing = train_test[(train_test.Pclass == 3)]

test["Fare"] = test["Fare"].fillna(fare_missing.Fare.mean())

train_test["Fare"] = train_test["Fare"].fillna(fare_missing.Fare.mean())

fig, ax = plt.subplots(figsize=(16,12),ncols=2,nrows=1)

ax1 = sns.boxplot(x="Survived", y="Age", hue="Pclass", data=train_test, ax = ax[0]);

ax2 = sns.boxplot(x="Pclass", y="Age", hue="Sex", data=train_test, ax = ax[1]);
test_na_age_index = list(test["Age"][test["Age"].isnull()].index)



for i in test_na_age_index:

    i_Pclass = test.iloc[i]["Pclass"]

    mean_pclass = test[test.Pclass==i_Pclass]['Age'].mean()

    age_pred = test[((test['Sex'] == test.iloc[i]["Sex"]) & (test['Pclass'] == test.iloc[i]["Pclass"]))]["Age"].mean()

    if not np.isnan(age_pred) :

        test['Age'].iloc[i] = age_pred

    else :

        test['Age'].iloc[i] = mean_pclass
train_na_age_index = list(train["Age"][train["Age"].isnull()].index)



for i in train_na_age_index:

    i_Pclass = train.iloc[i]["Pclass"]

    mean_pclass = train[train.Pclass==i_Pclass]['Age'].mean()

    age_pred = train[((train['Survived'] == train.iloc[i]["Survived"])&(train['Sex'] == train.iloc[i]["Sex"]) & (train['Pclass'] == train.iloc[i]["Pclass"]))]["Age"].mean()

    if not np.isnan(age_pred) :

        train['Age'].iloc[i] = age_pred

    else :

        train['Age'].iloc[i] = mean_pclass
train.drop(["Cabin"], axis = 1, inplace=True)

test.drop(["Cabin"], axis = 1, inplace=True)

train_test.drop(["Cabin"], axis = 1, inplace=True)
train['Sex'] = train['Sex'].map({"male": 0, "female": 1})

test['Sex'] = test['Sex'].map({"male": 0, "female": 1})
one_hot_train = pd.get_dummies(train["Embarked"], drop_first=True)

one_hot_test = pd.get_dummies(test["Embarked"], drop_first=True)

#This is called one hot encoding. It is same think with above cell. The reason behind dropping first column is effiency.

#If both value is 0, it means it is dropped column. 

one_hot_train
train.drop(["Embarked"], axis = 1, inplace=True)

train = train.join(one_hot_train)



test.drop(["Embarked"], axis = 1, inplace=True)

test = test.join(one_hot_test)

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in train["Name"]]

train["Title"] = pd.Series(dataset_title)



dataset_title2 = [i.split(",")[1].split(".")[0].strip() for i in test["Name"]]

test["Title"] = pd.Series(dataset_title2)



train["Title"] = train["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train["Title"] = train["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

train["Title"] = train["Title"].astype(int)



test["Title"] = test["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test["Title"] = test["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

test["Title"] = test["Title"].astype(int)
pd.cut(train['Age'], 6)
train.loc[ train['Age'] <= 13, 'Age'] = 0,

train.loc[(train['Age'] > 13) & (train['Age'] <= 27), 'Age'] = 1,

train.loc[(train['Age'] > 27) & (train['Age'] <= 40), 'Age'] = 2,

train.loc[(train['Age'] > 40) & (train['Age'] <= 53), 'Age'] = 3,

train.loc[(train['Age'] > 53) & (train['Age'] <= 66), 'Age'] = 4,

train.loc[ train['Age'] > 66, 'Age'] = 5

test.loc[ train['Age'] <= 13, 'Age'] = 0,

test.loc[(train['Age'] > 13) & (train['Age'] <= 27), 'Age'] = 1,

test.loc[(train['Age'] > 27) & (train['Age'] <= 40), 'Age'] = 2,

test.loc[(train['Age'] > 40) & (train['Age'] <= 53), 'Age'] = 3,

test.loc[(train['Age'] > 53) & (train['Age'] <= 66), 'Age'] = 4,

test.loc[ train['Age'] > 66, 'Age'] = 5



e = pd.get_dummies(train["FamilySize"], drop_first=True)

train.drop(["FamilySize"], axis = 1, inplace=True)

train = train.join(e)



f = pd.get_dummies(test["FamilySize"], drop_first=True)

test.drop(["FamilySize"], axis = 1, inplace=True)

test = test.join(f)
test.drop(["Name"], axis = 1, inplace=True)

train.drop(["Name"], axis = 1, inplace=True)

test.drop(["SibSp"], axis = 1, inplace=True)

train.drop(["SibSp"], axis = 1, inplace=True)

test.drop(["Parch"], axis = 1, inplace=True)

train.drop(["Parch"], axis = 1, inplace=True)

test.drop(["Ticket"], axis = 1, inplace=True)

train.drop(["Ticket"], axis = 1, inplace=True)

train.drop(["PassengerId"], axis = 1, inplace=True)

test.drop(["Fare"], axis = 1, inplace=True)

train.drop(["Fare"], axis = 1, inplace=True)
X_train = train.drop("Survived", axis=1)

y_train = train["Survived"]

X_test  = test.drop("PassengerId", axis=1).copy()
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(logreg.score(X_train, y_train))
r_forest = RandomForestClassifier(n_estimators=200)

r_forest.fit(X_train, y_train)

y_pred = r_forest.predict(X_test)

r_forest.score(X_train, y_train)



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

knn.score(X_train, y_train)

svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

svc.score(X_train, y_train)
d_tree = DecisionTreeClassifier()

d_tree.fit(X_train, y_train)

y_pred = d_tree.predict(X_test)

d_tree.score(X_train, y_train)
gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

y_pred = gaussian.predict(X_test)

gaussian.score(X_train, y_train)

ada=AdaBoostClassifier(n_estimators=200,learning_rate=0.1)

ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)

ada.score(X_train, y_train)

Submission = pd.DataFrame({ 'PassengerId': test["PassengerId"],

                            'Survived': y_pred })

Submission.to_csv("Submission1.csv", index=False)