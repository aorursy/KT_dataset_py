import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)

import numpy as np

import sklearn.ensemble as ske

from sklearn import tree

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn import svm, cross_validation
df_train = pd.read_csv('../input/train.csv', index_col=['PassengerId'])

df_test = pd.read_csv('../input/test.csv', index_col=['PassengerId'])



# merge train set into master set

df_full = df_train.copy()

df_full['Train'] = 1

df_full['Train'] = df_full.Train.astype(int)



# merge test set into master set

df_full = pd.concat([df_full, df_test])

df_full.loc[df_full.Train.isnull(), 'Train'] = 0



print("Training shape: ", df_train.shape)

print("Test shape: ", df_test.shape)

print("Full shape: ", df_full.shape)
# Print all column names and their count of NAN values

for col in df_full.columns:

    print("Found {} NAN values for column: {}".format(df_full[col].isnull().sum(), col))
# Handle missing data for attribute 'Embarked'

df_full.Embarked.value_counts().plot(kind='bar');

df_full['Surname'] = df_full.Name.str.replace('(,.*)', '')

# the bar plot of 'Embarked' shows, that the most people embarked from 'Southampton' 

# and therefore I assume, that 'S' is a good replacement for the two missing values.

df_full.loc[df_full.Embarked.isnull(), 'Embarked'] = 'S'
# Handle NAN values for column 'Age'

# A good strategy, which will be suggested by other Kaggle user, 

# is to derive the ages by the help of the 'Title', contained in the 'Name' field.

df_full["Title"] = df_full.Name.str.replace('(.*, )|(\\..*)', '')



# show Title distribution

df_full["Title"].value_counts().plot(kind='bar');
# how many rows exists, where Age is NaN and Title is given

print(df_full.loc[(df_full.Age.isnull()) & (df_full.Ticket.notnull()), :].Title.value_counts())
# how many rows exists, where Age is NaN and also Title is NaN

print(df_full.loc[(df_full.Age.isnull()) & (df_full.Ticket.isnull()), :].Title.value_counts())
# determine the mean Age grouped by Title

df_full.groupby(['Title']).Age.mean().astype(int)
# assign the mean Ages per Title to the missing records

df_full.loc[(df_full.Age.isnull()) & (df_full.Title == 'Mr'), 'Age'] = 32

df_full.loc[(df_full.Age.isnull()) & (df_full.Title == 'Ms'), 'Age'] = 28

df_full.loc[(df_full.Age.isnull()) & (df_full.Title == 'Mrs'), 'Age'] = 36

df_full.loc[(df_full.Age.isnull()) & (df_full.Title == 'Miss'), 'Age'] = 21

df_full.loc[(df_full.Age.isnull()) & (df_full.Title == 'Master'), 'Age'] = 5

df_full.loc[(df_full.Age.isnull()) & (df_full.Title == 'Dr'), 'Age'] = 43



# convert age column to int

df_full.Age = df_full.Age.astype(int)
# Handle missing data for attribute 'Fare'

# Since there is only one record missing, and this passenger (Id=1044) embarked in 'Southampton'

# we assign the mean Fare rate for this Port to the passenger

df_full.loc[1044, 'Fare'] = df_full.groupby('Embarked').Fare.mean()['S']
# Handle missing 'Cabin' data

df_full.loc[:, 'Deck'] = df_full.Cabin.str[0]



# Remove the Deck 'T'

df_full.loc[df_full.Deck == 'T', 'Deck'] = np.nan



for row in df_full.loc[df_full.Deck.notnull(), ['Deck', 'Surname']].itertuples():

    df_full.loc[(df_full.Deck.isnull()) & (df_full.Surname == row.Surname), 'Deck'] = row.Deck



lr_clf = LogisticRegression()

X_deck_train = df_full.loc[df_full.Deck.notnull(), ['Embarked', 'Fare', 'Pclass']]

X_deck_test = df_full.loc[df_full.Deck.isnull(), ['Embarked', 'Fare', 'Pclass']]

X_deck_train = pd.get_dummies(X_deck_train)

X_deck_test = pd.get_dummies(X_deck_test)

y_deck_train = df_full.loc[df_full.Deck.notnull(), ['Deck']]

lr_clf.fit(X_deck_train, y_deck_train)

y_deck_pred = lr_clf.predict(X_deck_test)

# Since a lot of data is missing for this attribute, we remove this column fully

#df_full.drop('Cabin', axis=1, inplace=True)

df_deck_pred = pd.DataFrame({

        "Deck": y_deck_pred

        }, index=X_deck_test.index)



df_full.loc[df_full.Deck.isnull(), 'Deck'] = df_deck_pred.Deck
X_deck_train.isnull().sum()
df_full.info()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))



# assess the 'Parch' attribute (Number of Parents/Children Aboard)

sns.countplot(x='Parch', hue='Survived', data=df_full, ax=ax1)



# assess the 'SibSp' releation (Number of Siblings/Spouses Aboard)

sns.countplot(x='SibSp', hue='Survived', data=df_full, ax=ax2)
df_full['Familiy_members'] = df_full.SibSp + df_full.Parch + 1



# assess the new, derived parameter 'Familiy_members'

sns.countplot(x='Familiy_members', hue='Survived', data=df_full);
df_full['Familiy_size'] = df_full.loc[:, 'Familiy_members'].map({1:'S', 2:'M', 3:'M', 4:'M', 5:'L', 6:'L', 7:'L', 8:'L', 9:'L', 10:'L', 11:'L'})

sns.countplot(x='Familiy_size', hue='Survived', data=df_full);


s_group = df_full.groupby('Surname').Surname.count() >= 2

df_full.loc[(df_full.Surname.isin(s_group[s_group == True].index)) & (df_full.Sex == 'female') & (df_full.Parch > 0) & (df_full.Age >= 18),'Mother'] = 1

df_full.loc[df_full.Mother.isnull(), 'Mother'] = 0



sns.countplot(x='Mother', hue='Survived', data=df_full[df_full.Mother == 1]);
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

df_full.loc[df_full.Survived == 1].groupby('Age').Age.count().hist(ax=ax1, bins=10, normed=1)

df_full.loc[df_full.Survived == 0].groupby('Age').Age.count().hist(ax=ax2, bins=100, normed=1)

ax1.set_title("Count of survivors")

ax1.set_xlabel("Age")

ax1.set_ylabel("Count")

ax2.set_title("Count of victims")

ax2.set_xlabel("Age")

ax2.set_ylabel("Count")
df_full.loc[df_full.Age <= 6, 'Age_group'] = 'B' # Baby

df_full.loc[(df_full.Age > 6) & (df_full.Age <= 12), 'Age_group'] = 'C' # Children

df_full.loc[(df_full.Age > 12) & (df_full.Age <= 17), 'Age_group'] = 'Y' # Youngster

df_full.loc[(df_full.Age > 17) & (df_full.Age <= 26), 'Age_group'] = 'S' # Student

df_full.loc[(df_full.Age > 26) & (df_full.Age <= 50), 'Age_group'] = 'A' # Adult

df_full.loc[(df_full.Age > 50), 'Age_group'] = 'P' # Pensionier



sns.countplot(x='Age_group', hue='Survived', data=df_full, order=['B','C','Y','S','A','P']);
#features = ['Age_group', 'Embarked', 'Fare', 'Sex', 'Pclass', 'Familiy_size', 'Mother', 'Deck']

features = ['Age', 'Fare', 'Sex', 'Pclass', 'Embarked', 'Familiy_size']



# copy only the relevant features from the master data set, to the prediction matrix.

X_train = df_full.loc[df_full.Train == 1, features].copy()



X_test = df_full.loc[df_full.Train == 0, features].copy()



print("Shape X Train: ", X_train.shape)

print("Shape X Test: ", X_test.shape)
# scale all categorical features

X_train = pd.get_dummies(X_train, drop_first=True)

X_test = pd.get_dummies(X_test, drop_first=True)



y_train = df_full.loc[df_full.Train == 1, 'Survived']



print("Shape X Train: ", X_train.info())

print("Shape X Test: ", X_test.info())
X_train.info()
#from sklearn.preprocessing import StandardScaler

#stdsc = StandardScaler()

#X_train_std = stdsc.fit_transform(X_train)

#X_test_std = stdsc.transform(X_test)



#print("Shape Std X Train: ", X_train_std.shape)

#print("Shape Std X Test: ", X_test_std.shape)
shuffle_validator = cross_validation.ShuffleSplit(len(X_train), n_iter=20, test_size=0.2, random_state=0)

def test_classifier(clf):

    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=shuffle_validator)

    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))
def save_result(y_pred, name):

    pd.DataFrame({

        "PassengerId": df_full.loc[df_full.Train == 0, :].index,

        "Survived": y_pred

        }).to_csv("predictions_{}.csv".format(name), index=False)
def train_predict_classifier(clf):

    clf.fit(X_train, y_train)

    return clf.predict(X_test)
def print_feature_importance(clf):

    imp_feat = clf.feature_importances_

    df_feat = pd.DataFrame({ 'feature' : X_test.columns, 'importance' : imp_feat})

    df_feat = df_feat.sort_values(by=['importance'], ascending=False)

    print(df_feat)
k_range = range(1, 40)

param_grid = dict(n_neighbors=list(k_range),weights = ["uniform", "distance"])
knn = KNeighborsClassifier(n_neighbors=5)

grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')

grid.fit(X_train, y_train)

print("Best score: ", grid.best_score_)

print("Best params: ", grid.best_params_)
lrcls = LogisticRegression()

test_classifier(lrcls)
svc = svm.SVC()

test_classifier(svc)
clf_dt = tree.DecisionTreeClassifier(max_depth=10)

test_classifier(clf_dt)

train_predict_classifier(clf_dt)

print_feature_importance(clf_dt)
clf_gb = ske.GradientBoostingClassifier(n_estimators=50)

test_classifier(clf_gb)

save_result(train_predict_classifier(clf_gb), "grad_clf")
clf_rf = ske.RandomForestClassifier(n_estimators=50)

test_classifier(clf_rf)

train_predict_classifier(clf_rf)

print_feature_importance(clf_rf)
eclf = ske.VotingClassifier([('dt', clf_dt), ('rf', clf_rf), ('gb', clf_gb)])

test_classifier(eclf)
save_result(train_predict_classifier(eclf), "vote_clf")