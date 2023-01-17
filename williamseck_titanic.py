import pandas as pd

import numpy as np

import pylab

import statistics

from statistics import mean

from matplotlib import pyplot as plt

import seaborn as sns

from array import array

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from lightgbm import LGBMClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from pandas import Series,DataFrame

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
train = pd.read_csv('../input/survivants-du-titanic/train.csv', sep = ',')

test = pd.read_csv('../input/survivants-du-titanic/test.csv', sep =',')

submission = pd.read_csv('../input/titanic/gender_submission.csv')

PassengerId = test['PassengerId']
train.head()
def plot_correlation_map( train ):

    corr = train.corr()

    _ , ax = plt.subplots( figsize =( 8 , 8 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )

plot_correlation_map(train)
def plot_cat(data, x_axis, y_axis, hue):

    plt.figure()    

    sns.barplot(x=x_axis, y=y_axis, hue=hue, data=data)

    sns.set_context("notebook", font_scale=1.6)

    plt.legend(loc="upper right", fontsize="medium")

plot_cat(train,"Sex", "Survived", None) 
plot_cat(train,"Pclass", "Survived", "Sex") 

plot_cat(train,"Pclass", "Survived", None)
# On observe qu'une grande majorité des victimes provient de la 3eme classe. (pas surprenant).

# En première classe, le ratio est inversé.
pd.isnull(train).sum() #données manquantes
train.head(2)
train.shape
# On index le dataframe avec la variable 'Passengerld' car elle n'apporte aucune information intéressante
train.set_index('PassengerId', inplace =True, drop =True )
train.columns
train.dtypes
train.count()
def parse_model_0(X):

    target = X.Survived

    X=X[['Fare','SibSp','Parch']] #(Valeurs ayant des données complètes)😊

    return X, target
X,y = parse_model_0(train.copy())
X.head(2)
def compute_score(clf, X, y):

    xval = cross_val_score(clf, X, y, cv = 5)

    return mean(xval) 

#le jeu de données étant très petit, le score varie beaucoup. On prend donc la moyenne)
lr = LogisticRegression()

compute_score( lr , X , y )
train1 = pd.get_dummies(train, columns=['Pclass'])
def parse_model_1(X):

    target = X.Survived

    X=X[['Fare','SibSp','Parch','Pclass_1','Pclass_2','Pclass_3']] 

    return X, target
X,y = parse_model_1(train1.copy())
def compute_score(clf, X, y):

    xval = cross_val_score(clf, X, y, cv = 5)

    return mean(xval) 

#le jeu de données étant très petit, le score varie beaucoup. On prend donc la moyenne)
X.head(2)
lr = LogisticRegression()

compute_score(lr,X,y)
lr = LogisticRegression()

lr.fit(X,y)

print (lr.coef_)
# 2 possibilités. On va donc binéariser le sex.
train2 = pd.get_dummies(train1, columns=['Sex'])
def parse_model_2(X):

    target = X.Survived

    X=X[['Fare','SibSp','Parch','Pclass_1','Pclass_2','Pclass_3','Sex_female','Sex_male']] #(Valeurs ayant des données complètes)😊

    return X, target
X,y = parse_model_2(train2.copy())
def compute_score(clf, X, y):

    xval = cross_val_score(clf, X, y, cv = 5)

    return mean(xval) 

#le jeu de données étant très petit, le score varie beaucoup. On prend donc la moyenne)
X.head(2)
lr = LogisticRegression()

compute_score(lr,X,y)
# On augmente de 10 points. Le Sex était donc une variable très importante ("Les femmes d'abord"😄)
# On va remplacer les valeurs manquantes par la médiane
train["Age"].fillna(train["Age"].median(), inplace=True)

# convert from float to int

train2['Age'] = train['Age'].astype(int)
def parse_model_3(X):

    target = X.Survived

    X = X[['Age','Fare','SibSp','Parch','Pclass_1','Pclass_2','Pclass_3','Sex_female','Sex_male']]

    return X, target
X,y = parse_model_3(train2.copy())
X.head(2)
def compute_score(clf, X, y):

    xval = cross_val_score(clf, X, y, cv = 5)

    return mean(xval) 

#le jeu de données étant très petit, le score varie beaucoup. On prend donc la moyenne)
lr = LogisticRegression()

compute_score(lr,X,y)
# La variable 'Age' a été mal utilisé. Elle fait baisser le score. Il pourrait être interessant de faire des catégories: 

#Poussins [0,8] - Benjamin[9,12] - Minimes[13,15] - Cadet[15,18] - Junior[19,35] - Senior[35,60] - Vétéran[61,90]
rf = RandomForestClassifier()

compute_score(rf,X,y)
# On pourra ajouter les autres variables:

#'Name': On peut s'interresser au rang social en donnant du poids au titre ((Dr, Major, Miss...))🧐

#'Cabin': Trop d'informations manquantes😢
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LGBMClassifier(n_estimators=200, max_depth=2)
clf.fit(X_train,y_train)
scores = cross_val_score(clf,X,y,scoring='f1', cv=5)

print('FScore')

print(np.mean(scores))

print(np.std(scores))
%%time

scores = cross_val_score(clf,X,y,scoring='recall', cv=5)

print('Recall')

print(np.mean(scores))

print(np.std(scores))
scores = cross_val_score(clf,X,y,scoring='precision', cv=5)

print('Precision')

print(np.mean(scores))

print(np.std(scores))
%%time

scores = cross_val_score(clf,X,y, cv=5)

print('Accuracy')

print(np.mean(scores))

print(np.std(scores))
scores = cross_val_score(clf,X,y,scoring='roc_auc', cv=5)

print('AUC')

print(np.mean(scores))

print(np.std(scores))
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred_rf = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, y_train)
# Support Vector Machines

svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

svc.score(X_train, y_train)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

knn.score(X_train, y_train)
# get Correlation Coefficient for each feature using Logistic Regression

coeff_train = DataFrame(train.columns.delete(0))

coeff_train.columns = ['Features']

coeff_train["Coefficient Estimate"] = pd.Series(logreg.coef_[0])



# preview

coeff_train