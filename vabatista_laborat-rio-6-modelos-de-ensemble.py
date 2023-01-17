import numpy as np

import pandas as pd

from sklearn.model_selection import KFold
dfTitanic = pd.read_csv('../input/lab6_train_no_nulls_no_outliers.csv')

dfTitanic.head(3)
dfTitanic['Sex'] = pd.factorize(dfTitanic['Sex'].values)[0]

dfTitanic['Pclass'] = pd.factorize(dfTitanic['Pclass'].values)[0]

dfTitanic['Embarked'] = pd.factorize(dfTitanic['Embarked'].values)[0]

dfTitanic['Ticket'] = pd.factorize(dfTitanic['Ticket'].values)[0]
dfTitanic.head()
y = dfTitanic['Survived'].values

#X = dfTitanic[['Age', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S', '1', '2', '3', 'female', 'male']].values

X = dfTitanic[['Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked', 'Pclass', 'Sex']].values

kf = KFold(n_splits=5, shuffle=True)
from sklearn.metrics import accuracy_score, roc_auc_score



def avalia_classificador(clf, kf, X, y, f_metrica):

    metrica = []

    for train, valid in kf.split(X,y):

        x_train = X[train]

        y_train = y[train]

        x_valid = X[valid]

        y_valid = y[valid]

        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_valid)

        metrica.append(f_metrica(y_valid, y_pred))

    return np.array(metrica).mean()
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier

X_n = StandardScaler().fit_transform(X)

X_n = PolynomialFeatures(2).fit_transform(X_n)



rf = RandomForestClassifier(n_estimators=200, max_features=8, max_depth=12)

media_acuracia = avalia_classificador(rf, kf, X_n, y, accuracy_score) 

print('Acurácia: ', media_acuracia)

media_auc = avalia_classificador(rf, kf, X, y, roc_auc_score) 

print('AUC: ', media_auc)
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()
media_acuracia = avalia_classificador(ada, kf, X, y, accuracy_score) 

print('Acurácia: ', media_acuracia)

media_auc = avalia_classificador(ada, kf, X, y, roc_auc_score) 

print('AUC: ', media_auc)
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
media_acuracia = avalia_classificador(gbc, kf, X, y, accuracy_score) 

print('Acurácia: ', media_acuracia)

media_auc = avalia_classificador(gbc, kf, X, y, roc_auc_score) 

print('AUC: ', media_auc)