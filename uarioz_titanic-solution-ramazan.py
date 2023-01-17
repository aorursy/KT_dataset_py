# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



print("Pandas Versiyon: ", pd.__version__)

print("Numpy Versiyon: ", np.__version__)

print("Matplotlib Versiyon: ", matplotlib.__version__)

print("Seaborn Versiyon: ", sns.__version__)

print("Python Versiyon: ", sys.version)



import warnings

warnings.filterwarnings('ignore')
train_set = pd.read_csv("../input/titanic/train.csv")

train_set.head()
test_set = pd.read_csv("../input/titanic/test.csv")

test_set.head()
print(train_set.info())

print(test_set.info())
print("Eğitim setinin boyutu: ", train_set.shape)

print("Test setinin boyutu: ", test_set.shape)
# Train seti için boş değerler



train_set.isnull().sum()
# Test seti için boş değerler



test_set.isnull().sum()
train_set.describe(include = 'all').T
test_set.describe(include = "all").T
# Kategorilerin hayatta kalma ortalamaları



train_set.groupby("Survived").mean()
# Hayatta kalanların cinsiyetlere göre oranları



train_set.groupby('Sex')[['Survived']].mean()
# Cinsiyetlere göre yaş ortalamaları



train_set.groupby('Sex')['Age'].mean()
# Cinsiyet ve Sınıfa göre hayatta kalma oranı



train_set.pivot_table('Survived', index = 'Sex', columns = 'Pclass')
# Sınıf ve yaş ölçeklendirmesine göre hayatta kalma oranları



age = pd.cut(train_set['Age'], [0, 18, 80])

train_set.pivot_table('Survived', ['Sex', age], 'Pclass')
# Cinsiyet ve Sınıfa göre hayatta kalma oranı



train_set.pivot_table('Survived', index = 'Sex', columns = 'Pclass').plot()
# Yaşlara göre hayatta kalma oranları



SurvAge = sns.FacetGrid(train_set, col="Survived")

SurvAge.map(plt.hist, 'Age', bins=20)
# Yolcuların sınıflarına göre hayatta kalma oranları



sns.barplot(x = 'Pclass', y = 'Survived', data = train_set)
# Yolcuların yaş ve sınıflarına göre hayatta kalma durumları



spca = sns.FacetGrid(train_set, col='Survived', row='Pclass', size=3, aspect=2)

spca.map(plt.hist, 'Age', alpha=1, bins=25)

spca.add_legend();
# Her sınıf için ödenen fiyat



plt.scatter(train_set['Fare'], train_set['Pclass'], color = 'blue', label = 'Passanger Paid')

plt.ylabel('PClass')

plt.xlabel('Price / Fare')

plt.title('Price of each class')

plt.show()
# Korelasyon analizi

# Korelasyon: İki rassal değişken arasındaki doğrusal ilişkinin yönünü ve gücünü belirtir



sns.heatmap(train_set.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
# Her kategorideki tüm değerlerin kaçar tane olduğu



for val in train_set:

    print(train_set[val].value_counts())

    print()
# Eksik değerlerin doldurulması



# Age

train_set['Age'].fillna(train_set['Age'].median(), inplace = True)

test_set['Age'].fillna(test_set['Age'].median(), inplace = True)



# Embarked

train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace = True)



# Fare

test_set['Fare'].fillna(test_set['Fare'].median(), inplace = True)

train_set.isna().sum()
column_name = ['Name', 'Cabin', 'Ticket']

pss = ['PassengerId']

train_set.drop(column_name, axis = 1, inplace = True)

train_set.drop(pss, axis = 1, inplace = True)

test_set.drop(column_name, axis = 1, inplace = True)
train_set.head()
test_set.head()
# Sex -> Label Encoding



from sklearn import preprocessing 



le = preprocessing.LabelEncoder() 



train_set['Sex'] = le.fit_transform(train_set['Sex'])

test_set['Sex'] = le.fit_transform(test_set['Sex'])

train_set.head()
# Embarked -> One-Hot Encoding



#train_set = pd.get_dummies(train_set, columns = ['Embarked'], drop_first = True)

#test_set = pd.get_dummies(test_set, columns = ['Embarked'], drop_first = True)



train_set["Embarked"] = train_set["Embarked"].map({"S":0, "Q":1, "C":2})

test_set["Embarked"] = test_set["Embarked"].map({"S":0, "Q":1, "C":2})
train_set.head(20)
from sklearn.preprocessing import StandardScaler



ss = StandardScaler()



train_set[['Age']] = ss.fit_transform(train_set[['Age']])

test_set[['Age']] = ss.transform(test_set[['Age']])
train_set.head()
# 0-7.91 -> 0, 7.91-14.45 -> 1, 14.45-31 -> 2, +31 -> 3



train_set.loc[train_set.Fare <= 7.91, 'Fare'] = 0

train_set.loc[(train_set.Fare > 7.91) & (train_set.Fare <= 14.45), 'Fare'] = 1

train_set.loc[(train_set.Fare > 14.45) & (train_set.Fare <= 31), 'Fare'] = 2

train_set.loc[(train_set.Fare > 31), 'Fare'] = 3
# 0-7.89 -> 0, 7.89-14.45 -> 1, 14.45-31.47 -> 2, +31.47 -> 3



test_set.loc[test_set.Fare <= 7.89, 'Fare'] = 0

test_set.loc[(test_set.Fare > 7.89) & (test_set.Fare <= 14.45), 'Fare'] = 1

test_set.loc[(test_set.Fare > 14.45) & (test_set.Fare <= 31.47), 'Fare'] = 2

test_set.loc[(test_set.Fare > 31.47), 'Fare'] = 3
train_set.tail()
# Veri setindeki yeni satır ve sütun sayısı



train_set.shape
# Test setindeki yeni satır ve sütun sayısı



test_set.shape
# Sütunların benzersiz değerleri



print(train_set['Age'].unique())

print(train_set['Fare'].unique())
train_set.head()
# Verilerin X ve Y değişkenlerine atanması. X_train -> Bağımsız, y_train -> Bağımlı



X_train = train_set.iloc[:, 1:8].values

y_train = train_set.iloc[:, 0].values

X_test = test_set.drop('PassengerId', axis = 1).values
print("X_train: ", X_train.shape)

print("X_test: ", X_test.shape)

print("y_train: ", y_train.shape)
# Makine öğrenmesi modelleri



# Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 0)

lr.fit(X_train, y_train)

lr_acc = round(lr.score(X_train, y_train) * 100, 2)

    

# KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

knn.fit(X_train, y_train)

knn_acc = round(knn.score(X_train, y_train) * 100, 2)

    

# SVM (Lin)

from sklearn.svm import SVC

svc_lin = SVC(kernel = 'linear', random_state = 0)

svc_lin.fit(X_train, y_train)

svc_lin_acc = round(svc_lin.score(X_train, y_train) * 100, 2)

    

# SVM (Rbf)

from sklearn.svm import SVC

svc_rbf = SVC(kernel = 'rbf', random_state = 0)

svc_rbf.fit(X_train, y_train)

svc_rbf_acc = round(svc_rbf.score(X_train, y_train) * 100, 2)

    

# SGDClassifier

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()

sgd.fit(X_train, y_train)

sgd_acc = round(sgd.score(X_train, y_train) * 100, 2)

    

# XGBClassifier

from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

xgb_acc = round(xgb.score(X_train, y_train) * 100, 2)

    

# GaussianNB

from sklearn.naive_bayes import GaussianNB

gauss = GaussianNB()

gauss.fit(X_train, y_train)

gauss_acc = round(gauss.score(X_train, y_train) * 100, 2)

    

# GaussianProcessClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

gauss_pro = GaussianProcessClassifier()

gauss_pro.fit(X_train, y_train)

gauss_pro_acc = round(gauss_pro.score(X_train, y_train) * 100, 2)

    

# DecisionTree

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

tree.fit(X_train, y_train)

tree_acc = round(tree.score(X_train, y_train) * 100, 2)

    

# RandomForest

from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)

randomforest.fit(X_train, y_train)

randomforest_acc = round(randomforest.score(X_train, y_train) * 100, 2)

    

# Perceptron

from sklearn.linear_model import Perceptron

prcp = Perceptron()

prcp.fit(X_train, y_train)

prcp_acc = round(prcp.score(X_train, y_train) * 100, 2)

    

# AdaBoostClassifier

from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()

ada.fit(X_train, y_train)

ada_acc = round(ada.score(X_train, y_train) * 100, 2)

    

# GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)

gbc_acc = round(gbc.score(X_train, y_train) * 100, 2)





# Her modelin eğitim doğruluğu

   

print('Logistic Regression Eğitim Doğruluğu: ', lr_acc)

print('KNN Eğitim Doğruluğu: ', knn_acc)

print('Linear SVC Eğitim Doğruluğu: ', svc_lin_acc)

print('RBF SVC Eğitim Doğruluğu: ', svc_rbf_acc)

print('SGD Eğitim Doğruluğu: ', sgd_acc)

print('XGBClassifier eğitim Doğruluğu: ', xgb_acc)

print('Naive Bayes Eğitim Doğruluğu: ', gauss_acc)

print('Gaussian Process Classifier Eğitim Doğruluğu: ', gauss_pro_acc)

print('Decision Tree Eğitim Doğruluğu: ', tree_acc)

print('Random Forests Eğitim Doğruluğu: ', randomforest_acc)

print('Perceptron Eğitim Doğruluğu: ', prcp_acc)

print('Ada Boost Classifier Eğitim Doğruluğu: ', ada_acc)

print('Gradient Boosting Classifier Eğitim Doğruluğu: ', gbc_acc)
y_pred = lr.predict(X_test)

y_pred
data = pd.read_csv("../input/titanic/gender_submission.csv")

Y = data['Survived']
from sklearn.metrics import confusion_matrix



f,ax = plt.subplots(4,4, figsize = (18, 10))



sns.heatmap(confusion_matrix(Y, lr.predict(X_test)), ax = ax[0,0], annot = True, fmt = '2.0f')

ax[0, 0].set_title('LogReg CM')



sns.heatmap(confusion_matrix(Y, knn.predict(X_test)), ax = ax[0, 1], annot = True, fmt = '2.0f')

ax[0, 1].set_title('KNN CM')



sns.heatmap(confusion_matrix(Y, svc_lin.predict(X_test)), ax = ax[0, 2], annot = True, fmt = '2.0f')

ax[0, 2].set_title('SVC_Lin CM')



sns.heatmap(confusion_matrix(Y, svc_rbf.predict(X_test)), ax = ax[0, 3], annot = True, fmt = '2.0f')

ax[0, 3].set_title('SVC_RBF CM')



sns.heatmap(confusion_matrix(Y, sgd.predict(X_test)), ax = ax[1, 0], annot = True, fmt = '2.0f')

ax[1, 0].set_title('SGD CM')



sns.heatmap(confusion_matrix(Y, xgb.predict(X_test)), ax = ax[1, 1], annot = True, fmt = '2.0f')

ax[1, 1].set_title('XGB CM')



sns.heatmap(confusion_matrix(Y, gauss.predict(X_test)), ax = ax[1, 2], annot = True, fmt = '2.0f')

ax[1, 2].set_title('Gauss CM')



sns.heatmap(confusion_matrix(Y, gauss_pro.predict(X_test)), ax = ax[1, 3], annot = True, fmt = '2.0f')

ax[1, 3].set_title('GPro CM')



sns.heatmap(confusion_matrix(Y, tree.predict(X_test)), ax = ax[2, 0], annot = True, fmt = '2.0f')

ax[2, 0].set_title('DT CM')



sns.heatmap(confusion_matrix(Y, randomforest.predict(X_test)), ax = ax[2, 1], annot = True, fmt = '2.0f')

ax[2, 1].set_title('RF CM')



sns.heatmap(confusion_matrix(Y, prcp.predict(X_test)), ax = ax[2, 2], annot = True, fmt = '2.0f')

ax[2, 2].set_title('Prcp CM')



sns.heatmap(confusion_matrix(Y, ada.predict(X_test)), ax = ax[2, 3], annot = True, fmt = '2.0f')

ax[2, 3].set_title('Ada CM')



sns.heatmap(confusion_matrix(Y, gbc.predict(X_test)), ax = ax[3, 0], annot = True, fmt = '2.0f')

ax[3, 0].set_title('GBC CM')
# Karmaşıklık Matrisi (Confusion Matrix)



TN, FP, FN, TP = confusion_matrix(Y, lr.predict(X_test)).ravel()

lr_test = (TP + TN) / (TP + TN + FN + FP)



TN, FP, FN, TP = confusion_matrix(Y, knn.predict(X_test)).ravel()

knn_test = (TP + TN) / (TP + TN + FN + FP)



TN, FP, FN, TP = confusion_matrix(Y, svc_lin.predict(X_test)).ravel()

svc_lin_test = (TP + TN) / (TP + TN + FN + FP)



TN, FP, FN, TP = confusion_matrix(Y, svc_rbf.predict(X_test)).ravel()

svc_rbf_test = (TP + TN) / (TP + TN + FN + FP)



TN, FP, FN, TP = confusion_matrix(Y, sgd.predict(X_test)).ravel()

sgd_test = (TP + TN) / (TP + TN + FN + FP)



TN, FP, FN, TP = confusion_matrix(Y, xgb.predict(X_test)).ravel()

xgb_test = (TP + TN) / (TP + TN + FN + FP)



TN, FP, FN, TP = confusion_matrix(Y, gauss.predict(X_test)).ravel()

gauss_test = (TP + TN) / (TP + TN + FN + FP)



TN, FP, FN, TP = confusion_matrix(Y, gauss_pro.predict(X_test)).ravel()

gauss_pro_test = (TP + TN) / (TP + TN + FN + FP)



TN, FP, FN, TP = confusion_matrix(Y, tree.predict(X_test)).ravel()

tree_test = (TP + TN) / (TP + TN + FN + FP)



TN, FP, FN, TP = confusion_matrix(Y, randomforest.predict(X_test)).ravel()

randomforest_test = (TP + TN) / (TP + TN + FN + FP)



TN, FP, FN, TP = confusion_matrix(Y, prcp.predict(X_test)).ravel()

prcp_test = (TP + TN) / (TP + TN + FN + FP)



TN, FP, FN, TP = confusion_matrix(Y, ada.predict(X_test)).ravel()

ada_test = (TP + TN) / (TP + TN + FN + FP)



TN, FP, FN, TP = confusion_matrix(Y, gbc.predict(X_test)).ravel()

gbc_test = (TP + TN) / (TP + TN + FN + FP)



print("Lojistic Regression Test Doğruluğu = ", lr_test)

print('KNN Test Doğruluğu: ', knn_test)

print('Linear SVC Test Doğruluğu: ', svc_lin_test)

print('RBF SVC Test Doğruluğu: ', svc_rbf_test)

print('SGD Test Doğruluğu: ', sgd_test)

print('XGBClassifier Test Doğruluğu: ', xgb_test)

print('Naive Bayes Test Doğruluğu: ', gauss_test)

print('Gaussian Process Classifier Test Doğruluğu: ', gauss_pro_test)

print('Decision Tree Test Doğruluğu: ', tree_test)

print('Random Forests Test Doğruluğu: ', randomforest_test)

print('Perceptron Test Doğruluğu: ', prcp_test)

print('Ada Boost Classifier Test Doğruluğu: ', ada_test)

print('Gradient Boosting Classifier Test Doğruluğu: ', gbc_test)
accuracy = {'Lojistik Regresyon':[lr_acc, lr_test * 100], 'KNN': [knn_acc, knn_test * 100],

           'Linear SVC': [svc_lin_acc, svc_lin_test * 100], 'RBF SVC': [svc_rbf_acc, svc_rbf_test * 100],

           'SGD': [sgd_acc, sgd_test * 100], 'XGB': [xgb_acc, xgb_test * 100], 'Naive Bayes': [gauss_acc, gauss_test * 100],

            'Process': [gauss_pro_acc, gauss_pro_test * 100], 'Decision': [tree_acc, tree_test * 100],

            'Random Forest': [randomforest_acc, randomforest_test * 100], 'Perceptron': [prcp_acc, prcp_test * 100],

            'Ada': [ada_acc, ada_test * 100], 'GBC': [gbc_acc, gbc_test * 100]}

accuracy
y_pred = xgb.predict(X_test)
df_submission = pd.read_csv("/kaggle/input/titanic/test.csv")



df_submission['Survived'] = y_pred



df_submission.drop(df_submission.columns.difference(['PassengerId', 'Survived']), axis=1, inplace=True)



df_submission.head(10)
df_submission.count()
df_submission.to_csv('submission.csv', index=False)