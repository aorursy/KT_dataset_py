# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 1) Import all Library that will be used
%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import linear_model, svm, gaussian_process
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# 1) Data treatment and cleaning

df_train_original = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test_original = pd.read_csv('/kaggle/input/titanic/test.csv')

df_train = df_train_original
df_test = df_test_original

df_train = df_train.drop(['Name'], axis=1)
df_test = df_test.drop(['Name'], axis=1)

df_train = df_train.drop(['Cabin'], axis=1)
df_test = df_test.drop(['Cabin'], axis=1)

df_train = df_train.drop(['Ticket'], axis=1)
df_test = df_test.drop(['Ticket'], axis=1)

df_train = df_train.drop(['SibSp'], axis=1)
df_test = df_test.drop(['SibSp'], axis=1)

df_train = df_train.drop(['Parch'], axis=1)
df_test = df_test.drop(['Parch'], axis=1)

#concatena os dados do treino e teste, apenas entres os campos "Pclass" e "Embarked"
# Ou seja, o campo "PassengerId" de ambos DFs serão deletados e o campo "Survived" do DF Treino

all_data = pd.concat((df_train.loc[:,'Sex':'Fare'],
                      df_test.loc[:,'Sex':'Fare']))


# Get_Dummies para transformar categoricos em Numéricos 
all_data = pd.get_dummies(all_data)

# Substitui os campos nulos pelas médias da coluna em questão
all_data = all_data.fillna(all_data.mean())
#all_data = all_data.fillna(0)

#creating matrices for sklearn:

#Cria Matriz X_train utilizando a Matriz com todos os dados all_data: do inicio da matriz (:) até o fim  da matriz df_train.shape[0]
X_train = all_data[:df_train.shape[0]]

#Cria Matriz X_test utilizando a Matriz com todos os dados all_data: a partir do último registro matriz df_train.shape[0], ou seja, todos os registros que não estiverem em df_train
X_test = all_data[df_train.shape[0]:]

# Cria o y, ou seja, o que será previsto, apenas com o campo "Survived"
y = df_train.Survived

# 2) Aplly Gradient Boost Model

from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.formula.api as smf
from sklearn.preprocessing import scale
gbr = GradientBoostingRegressor()

gbr.fit(X_train, y)

yhat_Train = gbr.predict(X_train)
#yhat_Train

yhat_test = gbr.predict(X_test)
#yhat_test

yhat_rounded = [round(x,ndigits=None) for x in yhat_test]
yhat_rounded = [int(x) for x in yhat_rounded]

yhat_gbr = yhat_rounded
print ('# # # # Esse é o yhat com o método Gradiente Descendente # # # #')
print ('# # # # Ou seja, a previsão se Esse é o yhat com o método Gradiente Descendente # # # #')
print (yhat_gbr)

# Gerando um CSV para o resultado obtido com o Gradiente Descendente:
df_test_gbr = df_test
df_test_gbr['Survived'] = yhat_gbr
df_test_gbr = df_test_gbr.drop(['Pclass'], axis=1)
df_test_gbr = df_test_gbr.drop(['Sex'], axis=1)
df_test_gbr = df_test_gbr.drop(['Age'], axis=1)
df_test_gbr = df_test_gbr.drop(['Fare'], axis=1)
df_test_gbr = df_test_gbr.drop(['Embarked'], axis=1)
df_test_gbr.to_csv('Titanic_GBR.csv', index = False)
# 3) Aplly Logistic Regression Model
from sklearn.linear_model import LogisticRegression

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y)

# accuracy = round(logreg.score(X_train, y) * 100, 2)
# print(accuracy)

LR_yhat_train = logreg.predict(X_train)

LR_yhat_test = logreg.predict(X_test)

print ('# # # # Esse é o yhat com o método Regressão Logistica # # # #')
print (LR_yhat_test)

# Gerando um CSV para o resultado obtido com Regressão Logistica:
df_test_RL = df_test
df_test_RL['Survived'] = LR_yhat_test
df_test_RL = df_test_RL.drop(['Pclass'], axis=1)
df_test_RL = df_test_RL.drop(['Sex'], axis=1)
df_test_RL = df_test_RL.drop(['Age'], axis=1)
df_test_RL = df_test_RL.drop(['Fare'], axis=1)
df_test_RL = df_test_RL.drop(['Embarked'], axis=1)
df_test_RL.to_csv('Titanic_RL.csv', index = False)
# 4) Aplly Random Forest Model

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y)

random_forest_train = random_forest.predict(X_train)
random_forest_test = random_forest.predict(X_test)

print ('# # # # Esse é o yhat com o método Random Forest # # # #')
print (random_forest_test)

# Gerando um CSV para o resultado obtido como o Randon Forest:
df_test_RF = df_test
df_test_RF['Survived'] = random_forest_test
df_test_RF = df_test_RF.drop(['Pclass'], axis=1)
df_test_RF = df_test_RF.drop(['Sex'], axis=1)
df_test_RF = df_test_RF.drop(['Age'], axis=1)
#df_test_RF = df_test_RF.drop(['SibSp'], axis=1)
#df_test_RF = df_test_RF.drop(['Parch'], axis=1)
#df_test_RF = df_test_RF.drop(['Ticket'], axis=1)
df_test_RF = df_test_RF.drop(['Fare'], axis=1)
#df_test_RF = df_test_RF.drop(['Cabin'], axis=1)
df_test_RF = df_test_RF.drop(['Embarked'], axis=1)
df_test_RF.to_csv('Titanic_RandonForest.csv', index = False)
# 5) Aplly XGBOOST Model

from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y)

xgb_train = xgb.predict(X_train)
xgb_test = xgb.predict(X_test)

print ('# # # # Esse é o yhat com o método Xgboost # # # #')
print (xgb_test)

# Gerando um CSV para o resultado obtido como o XGBOOST:
df_test_xgb = df_test
df_test_xgb['Survived'] = xgb_test
df_test_xgb = df_test_xgb.drop(['Pclass'], axis=1)
df_test_xgb = df_test_xgb.drop(['Sex'], axis=1)
df_test_xgb = df_test_xgb.drop(['Age'], axis=1)
#df_test_xgb = df_test_xgb.drop(['SibSp'], axis=1)
#df_test_xgb = df_test_xgb.drop(['Parch'], axis=1)
#df_test_xgb = df_test_xgb.drop(['Ticket'], axis=1)
df_test_xgb = df_test_xgb.drop(['Fare'], axis=1)
#df_test_xgb = df_test_xgb.drop(['Cabin'], axis=1)
df_test_xgb = df_test_xgb.drop(['Embarked'], axis=1)
df_test_xgb.to_csv('Titanic_Xgboost.csv', index = False)
# 6) Aplly KNeighbors Model

knn = KNeighborsClassifier()
knn.fit(X_train, y)

knn_train = knn.predict(X_train)
knn_test = knn.predict(X_test)

print ('# # # # Esse é o yhat com o método KNeighbors # # # #')
print (knn_test)

# Gerando um CSV para o resultado obtido como o KNeighbors:
df_test_knn = df_test
df_test_knn['Survived'] = knn_test
df_test_knn = df_test_knn.drop(['Pclass'], axis=1)
df_test_knn = df_test_knn.drop(['Sex'], axis=1)
df_test_knn = df_test_knn.drop(['Age'], axis=1)
#df_test_knn = df_test_knn.drop(['SibSp'], axis=1)
#df_test_knn = df_test_knn.drop(['Parch'], axis=1)
#df_test_knn = df_test_knn.drop(['Ticket'], axis=1)
df_test_knn = df_test_knn.drop(['Fare'], axis=1)
#df_test_knn = df_test_knn.drop(['Cabin'], axis=1)
df_test_knn = df_test_knn.drop(['Embarked'], axis=1)
df_test_knn.to_csv('Titanic_Knn.csv', index = False)
# 7) Aplly SVC Model

svc = SVC(probability=True)
svc.fit(X_train, y)

svc_train = svc.predict(X_train)
svc_test = svc.predict(X_test)

print ('# # # # Esse é o yhat com o método KNeighbors # # # #')
print (svc_test)

# Gerando um CSV para o resultado obtido como o SVC:
df_test_svc = df_test
df_test_svc['Survived'] = svc_test
df_test_svc = df_test_svc.drop(['Pclass'], axis=1)
df_test_svc = df_test_svc.drop(['Sex'], axis=1)
df_test_svc = df_test_svc.drop(['Age'], axis=1)
#df_test_svc = df_test_svc.drop(['SibSp'], axis=1)
#df_test_svc = df_test_svc.drop(['Parch'], axis=1)
#df_test_svc = df_test_svc.drop(['Ticket'], axis=1)
df_test_svc = df_test_svc.drop(['Fare'], axis=1)
#df_test_svc = df_test_svc.drop(['Cabin'], axis=1)
df_test_svc = df_test_svc.drop(['Embarked'], axis=1)
df_test_svc.to_csv('Titanic_SVC.csv', index = False)
# 8) Aplly Decision Tree Model

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y)

dtc_train = dtc.predict(X_train)
dtc_test = dtc.predict(X_test)

print ('# # # # Esse é o yhat com o método DecisionTree # # # #')
print (dtc_test)

# Gerando um CSV para o resultado obtido como o SVC:
df_test_dtc = df_test
df_test_dtc['Survived'] = dtc_test
df_test_dtc = df_test_dtc.drop(['Pclass'], axis=1)
df_test_dtc = df_test_dtc.drop(['Sex'], axis=1)
df_test_dtc = df_test_dtc.drop(['Age'], axis=1)
#df_test_dtc = df_test_dtc.drop(['SibSp'], axis=1)
#df_test_dtc = df_test_dtc.drop(['Parch'], axis=1)
#df_test_dtc = df_test_dtc.drop(['Ticket'], axis=1)
df_test_dtc = df_test_dtc.drop(['Fare'], axis=1)
#df_test_dtc = df_test_dtc.drop(['Cabin'], axis=1)
df_test_dtc = df_test_dtc.drop(['Embarked'], axis=1)
df_test_dtc.to_csv('Titanic_dtc.csv', index = False)
# 9) Aplly GaussianNB Model

gnb = GaussianNB()
gnb.fit(X_train, y)

gnb_train = gnb.predict(X_train)
gnb_test = gnb.predict(X_test)

print ('# # # # Esse é o yhat com o método GaussianNB # # # #')
print (gnb_test)

# Gerando um CSV para o resultado obtido como o SVC:
df_test_gnb = df_test
df_test_gnb['Survived'] = gnb_test
df_test_gnb = df_test_gnb.drop(['Pclass'], axis=1)
df_test_gnb = df_test_gnb.drop(['Sex'], axis=1)
df_test_gnb = df_test_gnb.drop(['Age'], axis=1)
#df_test_gnb = df_test_gnb.drop(['SibSp'], axis=1)
#df_test_gnb = df_test_gnb.drop(['Parch'], axis=1)
#df_test_gnb = df_test_gnb.drop(['Ticket'], axis=1)
df_test_gnb = df_test_gnb.drop(['Fare'], axis=1)
#df_test_gnb = df_test_gnb.drop(['Cabin'], axis=1)
df_test_gnb = df_test_gnb.drop(['Embarked'], axis=1)
df_test_gnb.to_csv('Titanic_gnb.csv', index = False)
