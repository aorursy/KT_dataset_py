
import numpy as np # linear algebra
from numpy import NaN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
%matplotlib inline
import glob
import missingno as mssno
seed=45
import random as rnd

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
import gc
import lightgbm as lb

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix, roc_auc_score ,roc_curve,auc,precision_recall_curve
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from scipy import stats, integrate
train=pd.read_csv("../input/train.csv",sep=',')
test=pd.read_csv("../input/test.csv",sep=',')
#Displaying Data
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()
#Missing values visualization
mssno.bar(train,color='g',figsize=(16,5),fontsize=12)
#Missing values visualization
mssno.bar(test,color='r',figsize=(16,5),fontsize=12)
train['Embarked'].value_counts()
train[train.Embarked.isnull()]
ax = sns.boxplot("Embarked","Fare", palette='rainbow', hue='Pclass',data=train)
plt.show()
ax = sns.boxplot("Embarked","Fare", palette='rainbow', hue='Pclass',data=test)
plt.show()
train.Embarked.fillna("C", inplace=True)
train.isnull().sum()
#for fare feature in test, lets take the mean
test['Fare'].mean()
test.Fare.fillna(test['Fare'].mean(),inplace=True)
test.Fare.isnull().sum()
plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Sex", 
            y = "Survived",
            data=train, 
            palette = "hls",
            linewidth=2 )
plt.title("Survived-Non-Survived Passenger Gender Distribution according to male, female", fontsize = 25)
plt.ylabel("% of passenger survived", fontsize = 15)
plt.xlabel("Sex",fontsize = 15);
plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Sex", 
            y = "Survived",
            hue="Pclass",
            data=train, 
            palette = "dark",
            linewidth=2 )
plt.title("Survived/Non-Survived Passenger Gender Distribution according to male, female and Pclass", fontsize = 25)
plt.ylabel("% of passenger survived", fontsize = 15)
plt.xlabel("Sex",fontsize = 15);
plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Sex", 
            y = "Survived",
            hue="Parch",
            data=train, 
            palette = "hls",
            linewidth=2 )
plt.title("Survived/Non-Survived Passenger Gender Distribution according to male, female and Parch", fontsize = 25)
plt.ylabel("% of passenger survived", fontsize = 15)
plt.xlabel("Sex",fontsize = 15);
a=train.groupby('Sex')['Survived'].value_counts()
a
a.plot(kind='bar',figsize=(15,8))
plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Pclass", 
            y = "Survived",
            data=train, 
            palette = "hls",
            linewidth=2 ),
plt.title("% of passengers survived according to class", fontsize = 25)
plt.ylabel("% of passenger survived", fontsize = 15)
plt.xlabel("Class",fontsize = 15);
fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Fare'],color='r',shade=True)
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'],color='k',shade=True)
plt.title("frequency of psngrs survived or not according to fare", fontsize = 25)
plt.ylabel("frequency of passenger", fontsize = 15)
plt.xlabel("Sex",fontsize = 15);
fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'],color='b',shade=True)
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Age'],color='m',shade=True)
plt.title("frequency of psngrs survived or not according to fare", fontsize = 25)
plt.ylabel("frequency of passenger", fontsize = 15)
plt.xlabel("Age",fontsize = 15);
fig = plt.figure(figsize=(15,8),)
a=train.loc[(train['Survived'] == 0),'Fare']
sns.distplot(a,color='brown');
fig = plt.figure(figsize=(15,8),)
a=train.loc[(train['Survived'] == 1),'Fare']
sns.distplot(a,color='brown');
g = sns.FacetGrid(train, size=8,hue="Survived", col ="Sex", margin_titles=True,
                palette='husl',)
g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
g.fig.suptitle("Survived by Sex, Fare and Age", size = 25)
plt.subplots_adjust(top=0.85)
g = sns.FacetGrid(train, col='Survived',size=8)
g.map(plt.hist, 'Age', bins=20,color='m')
g.fig.suptitle("Survived by Age", size = 25)
plt.subplots_adjust(top=0.85)
g = sns.FacetGrid(train, col='Survived',size=8)
g.map(plt.hist, 'Fare', bins=20,color='y')
g.fig.suptitle("Survived by Fare", size = 25)
plt.subplots_adjust(top=0.85)
g = sns.FacetGrid(train, size=12,hue="Survived", col ="Embarked", margin_titles=True,
                palette='dark',aspect=0.5)
g.map(plt.scatter, "Fare", "Age",edgecolor="g").add_legend()
g.fig.suptitle("Survived by Embarked, Fare and Age", size = 25)
plt.subplots_adjust(top=0.85)
sns.factorplot(x =  "SibSp", y = "Survived", data = train,kind = "point",size = 8,color='red')
plt.title('Factorplot of Sibilings and Spouses survived', fontsize = 25)
plt.subplots_adjust(top=0.85)
combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop(['index', 'PassengerId'], inplace=True, axis=1)
#Extracting  TITLE
combined_title = [i.split(",")[1].split(".")[0].strip() for i in combined["Name"]]
combined["Title"] = pd.Series(combined_title)
combined["Title"].head()
# Convert to categorical values Title 
combined["Title"] = combined["Title"].replace(['Capt', 'Col', 'Dr', 'Major', 'Rev'], 'officer')
combined["Title"] = combined["Title"].replace(['Lady', 'the Countess','Countess', 'Sir', 'Jonkheer', 'Dona','Don'], 'Royalty')
combined["Title"] = combined["Title"].replace(['Miss','Mlle'], 'Miss')
combined["Title"] = combined["Title"].replace(['Mrs','Ms','Mme'], 'Mrs')
combined["Title"] = combined["Title"].map({"Master":0, "Miss":1, "Mrs":2, "Mr":3, "officer":4,"Royalty":5})
combined["Title"] = combined["Title"].astype(int)
combined['family_size'] = combined.SibSp + combined.Parch+1
combined['Single'] = combined['family_size'].map(lambda s: 1 if s == 1 else 0)
combined['SmallF'] = combined['family_size'].map(lambda s: 1 if  s == 2  else 0)
combined['MedF'] = combined['family_size'].map(lambda s: 1 if 3 <= s <= 4 else 0)
combined['LargeF'] = combined['family_size'].map(lambda s: 1 if s >= 5 else 0)
combined['Sex'] = combined['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
combined.head()
combined['Embarked'] = combined['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )
combined.head()
Embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
combined = pd.concat([combined, Embarked_dummies], axis=1)
combined.drop('Embarked', inplace=True, axis=1)
front = train['Age']
train.drop(labels=['Age'], axis=1,inplace = True)
train.insert(0, 'Age', front)
train.head()
front = test['Age']
test.drop(labels=['Age'], axis=1,inplace = True)
test.insert(0, 'Age', front)
test.head()
guess_ages = np.zeros((2,3))
guess_ages
for i in range(0, 2):
    for j in range(0, 3):
        guess_df = combined[(combined['Sex'] == i) &  
                                 (combined['Pclass'] == j+1)]['Age'].dropna().astype(int)
            
            

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

        age_guess = guess_df.median()
            

            # Convert random age float to nearest .5 age
        guess_ages[i,j] =int( age_guess/0.5 + 0.5 ) * 0.5
            
for i in range(0, 2):
    for j in range(0, 3):
        combined.loc[ (combined.Age.isnull()) & (combined.Sex == i) & (combined.Pclass == j+1),\
                'Age'] = guess_ages[i,j]

combined['Age'] = combined['Age'].astype(int)

train.head()
combined.isnull().sum()
train['AgeBand'] = pd.cut(train['Age'], 5)
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
  
combined.loc[ combined['Age'] <= 16, 'Age'] = 0
combined.loc[(combined['Age'] > 16) & (combined['Age'] <= 32), 'Age'] = 1
combined.loc[(combined['Age'] > 32) & (combined['Age'] <= 48), 'Age'] = 2
combined.loc[(combined['Age'] > 48) & (combined['Age'] <= 64), 'Age'] = 3
combined.loc[ combined['Age'] > 64]
combined.head()
train.drop(['AgeBand'], axis=1)
combined.head()
combined["Cabin"].isnull().sum()
combined["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in combined['Cabin'] ])
combined.head()
cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
combined = pd.concat([combined, cabin_dummies], axis=1)

def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else: 
        return 'XXX'
tickets = set()
for t in combined['Ticket']:
    tickets.add(cleanTicket(t))
print( len(tickets))
def process_ticket():
    
    global combined
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip(), ticket)
        ticket = filter(lambda t : not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    

    # Extracting dummy variables from tickets:

combined['Ticket'] = combined['Ticket'].map(cleanTicket)
tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
combined = pd.concat([combined, tickets_dummies], axis=1)
combined.drop('Ticket', inplace=True, axis=1)
combined.head()
combined.drop('Name', inplace=True, axis=1)
combined.drop('SibSp', inplace=True, axis=1)
combined.drop('family_size', inplace=True, axis=1)
combined.drop('Cabin', inplace=True, axis=1)
title_dummies = pd.get_dummies(combined['Title'], prefix='Title')
combined = pd.concat([combined, title_dummies], axis=1)
combined.drop('Title', inplace=True, axis=1)
combined.head()
train1 = combined[:891]
test1 = combined[891:]
test1.drop(labels=["Survived"],axis = 1,inplace=True)
train1["Survived"] = train1["Survived"].astype(int)

Y_train = train1["Survived"]

X_train = train1.drop(labels = ["Survived"],axis = 1)

train1 = combined[:891]
test1 = combined[891:]
test1.drop(labels=["Survived"],axis = 1,inplace=True)
train1["Survived"] = train1["Survived"].astype(int)

Y_train = train1["Survived"]

X_train = train1.drop(labels = ["Survived"],axis = 1)
clf = RandomForestClassifier(n_estimators=25, max_features='sqrt')
clf = clf.fit(X_train, Y_train)

features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(25, 25))
model = SelectFromModel(clf, prefit=True)
X_train_reduced = model.transform(X_train)
print (X_train_reduced.shape)
test1_reduced = model.transform(test1)
print (test1_reduced.shape)
logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()

models = [logreg, logreg_cv, rf, gboost]
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)
for model in models:
    print ('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=X_train, y=Y_train, scoring='accuracy')
    print ('CV score = {0}'.format(score))
    print ('****')
parameters = {'bootstrap': False, 'min_samples_leaf': 4, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 5}
    
model = RandomForestClassifier(**parameters)
model.fit(X_train_reduced, Y_train)
output = model.predict(test1_reduced).astype(int)
model1 = round(model.score(X_train_reduced, Y_train) * 100, 2)
model1
output = model.predict(test1_reduced).astype(int)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived":output
    })
submission.to_csv("titanic51_submission.csv", index=False)