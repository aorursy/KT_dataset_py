# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')
#import the libraries for data cleaning and data visualising

import numpy as np

import scipy as sp

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import operator

warnings.filterwarnings('ignore')
#a quick look at the data

data.describe()
data.head()
print(data.columns)
print(data.shape)
#Finding the null count of each feature and removing the features with less data

data.isnull().sum()
data.shape
#Removing the feature which contain more null values

data = data.drop(columns = ['Sunshine','Evaporation','Cloud3pm','Cloud9am'], axis = 1)
data.shape
#let's see the correlation between all the columns

data_correlate = data.corr()

plt.figure(figsize = (15,15))

sns.heatmap(data_correlate, linewidth = 3, linecolor = 'black')

plt.show()
#Risk_MM can influnence the output. So droping that column too and date and location too.

data = data.drop(columns = ['Date', 'Location', 'RISK_MM'])
#Now looking at the columns

print(data.columns)
#Converting the RainToday and RainTomorrow data into binary

data['RainToday'].replace('No', 0, inplace = True)

data['RainToday'].replace('Yes', 1, inplace = True)

data['RainTomorrow'].replace('No', 0, inplace = True)

data['RainTomorrow'].replace('Yes', 1, inplace = True)

#Removing all the null values 

data = data.dropna(axis = 0)
data.shape
#Some columns needs to be changed into numeical values as they are categorical

non_numerical = ['WindGustDir', 'WindDir3pm', 'WindDir9am']

data = pd.get_dummies(data, columns= non_numerical)

rain_index = data.columns.get_loc('RainTomorrow')
data_1 = data
#standardizing the data.

from sklearn import preprocessing

standardizing = preprocessing.MinMaxScaler()

standardizing.fit(data_1)

data_standard = standardizing.transform(data_1)

data_standard = pd.DataFrame(data_standard, index=data_1.index, columns=data_1.columns)

data_standard.head()
#now on of the most important part. Feature selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

X = data_standard.loc[:,data_standard.columns!='RainTomorrow']

Y = data_standard[['RainTomorrow']]

selector = SelectKBest(chi2, k=3)

selector.fit(X, Y)

X_new = selector.transform(X)

print(X.columns[selector.get_support(indices=True)]) #top 3 columns
#the classification data:

data_classification = data_standard[['Rainfall', 'Humidity3pm', 'RainToday', 'RainTomorrow']]
#Spliting up the data:

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(data_standard[['Rainfall', 'Humidity3pm', 'RainToday']],data_standard['RainTomorrow'],test_size=0.25)
#Now let sstart analysing different models:
#1. Adaboost classifier:

from sklearn.ensemble import AdaBoostClassifier

adaboost_classifier = AdaBoostClassifier(n_estimators = 100)

adaboost_classifier.fit(X_train, y_train)

adaboost_classifier_pred = adaboost_classifier.predict(X_test)
from sklearn.metrics import accuracy_score

adaboost_classifier_score = accuracy_score(y_test, adaboost_classifier_pred)

print('The accuracy score obtained by adaboost classifier is:', adaboost_classifier_score)
#Bagging Classifer

from sklearn.ensemble import BaggingClassifier

bagging_classifier = BaggingClassifier()

bagging_classifier.fit(X_train, y_train)

bagging_classifier_pred = bagging_classifier.predict(X_test)

bagging_classifier_score = accuracy_score(y_test, bagging_classifier_pred)

print('the accuracy score of this bagging classifier is:', bagging_classifier_score)
#3. Extra Trees classifiers

from sklearn.ensemble import ExtraTreesClassifier

extra_trees_classifier = ExtraTreesClassifier(n_estimators = 100)

extra_trees_classifier.fit(X_train, y_train)

extra_trees_classifier_pred = extra_trees_classifier.predict(X_test)

extra_trees_classifier_score = accuracy_score(y_test, extra_trees_classifier_pred)

print('The accuracy obtained from extra trees classifiers is:', extra_trees_classifier_score)

#4. Gradient boosting classifier:

from sklearn.ensemble import GradientBoostingClassifier

gradient_boosting_classifier = GradientBoostingClassifier()

gradient_boosting_classifier.fit(X_train, y_train)

gradient_boosting_classifier_pred = gradient_boosting_classifier.predict(X_test)

gradient_boosting_classifier_score = accuracy_score(y_test, gradient_boosting_classifier_pred)

print('The accuracy score obtained from gradient boosting classifier:', gradient_boosting_classifier_score)
#5.Random forest classifier:

from sklearn.ensemble import RandomForestClassifier

random_forest_classifier = RandomForestClassifier()

random_forest_classifier.fit(X_train, y_train)

random_forest_classifier_pred = random_forest_classifier.predict(X_test)

random_forest_classifier_score = accuracy_score(y_test, random_forest_classifier_pred)

print('the accuracy score obtain from random forest classifier is :', random_forest_classifier_score)
#6 logistic regression;

from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()

logistic_regression.fit(X_train, y_train)

logistic_regression_pred = logistic_regression.predict(X_test)

logistic_regression_score = accuracy_score(y_test, logistic_regression_pred)

print('The accuracy score obtained from logistic regression is :', logistic_regression_score)
#7 passive aggressive classifier:

from sklearn.linear_model import PassiveAggressiveClassifier

passive_aggressive_classifier = PassiveAggressiveClassifier(C = 1.0, fit_intercept = True, max_iter = 1000)

passive_aggressive_classifier.fit(X_train, y_train)

passive_aggressive_classifier_pred = passive_aggressive_classifier.predict(X_test)

passive_aggressive_classifier_score = accuracy_score(y_test, passive_aggressive_classifier_pred)

print('The accuracy score is:', passive_aggressive_classifier_score)
#8.Ridge classifier;

# first using ridgeclassifierCV to find the alpha value

from sklearn.linear_model import RidgeClassifierCV

ridge_classifier_cv = RidgeClassifierCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), fit_intercept=True,

                                        normalize=False, scoring=None, cv=None,

                                        class_weight=None, store_cv_values=False)

ridge_classifier_cv.fit(X_train, y_train)

ridge_alpha = ridge_classifier_cv.alpha_

print('Obtained alpha value:', ridge_alpha)
from sklearn.linear_model import RidgeClassifier

ridge_classifier = RidgeClassifier(alpha = ridge_alpha, fit_intercept = True)

ridge_classifier.fit(X_train, y_train)

ridge_classifier_pred = ridge_classifier.predict(X_test)

ridge_classifier_score = accuracy_score(y_test, ridge_classifier_pred)

print('The accuracy score obtained is:', ridge_classifier_score)
#9. SGD classifier:

from sklearn.linear_model import SGDClassifier

sgd_classifier = SGDClassifier(alpha=0.0001,

                               l1_ratio=0.15, fit_intercept=True, max_iter=1000,

                               tol=0.001, shuffle=True)

sgd_classifier.fit(X_train, y_train)

sgd_classifier_pred = sgd_classifier.predict(X_test)

sgd_classifier_score = accuracy_score(y_test, sgd_classifier_pred)

print('the accuracy score calculated is:', sgd_classifier_score)
#10. Bernoulli's Naive-Bayes:

from sklearn.naive_bayes import BernoulliNB

bernoulli_nb = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True,class_prior=None)

bernoulli_nb.fit(X_train, y_train)

bernoulli_nb_pred = bernoulli_nb.predict(X_test)

bernoulli_nb_score = accuracy_score(y_test, bernoulli_nb_pred)

print('the score obtained is:', bernoulli_nb_score)
#11. Gaussian's Naive-Bayes:

from sklearn.naive_bayes import GaussianNB

gaussian_nb = GaussianNB(var_smoothing=1e-09)

gaussian_nb.fit(X_train, y_train)

gaussian_nb_pred = gaussian_nb.predict(X_test)

gaussian_nb_score = accuracy_score(y_test, gaussian_nb_pred)

print('the score obtained is:', gaussian_nb_score)
#12. Multinomial Naive-Bayes:

from sklearn.naive_bayes import MultinomialNB

multinomial_nb =  MultinomialNB()

multinomial_nb.fit(X_train, y_train)

multinomial_nb_pred = multinomial_nb.predict(X_test)

multinomial_nb_score = accuracy_score(y_test, multinomial_nb_pred)

print('the score obtained is:', multinomial_nb_score)
#13.complement naive bayes:

from sklearn.naive_bayes import ComplementNB

complement_nb =  ComplementNB()

complement_nb.fit(X_train, y_train)

complement_nb_pred = complement_nb.predict(X_test)

complement_nb_score = accuracy_score(y_test, complement_nb_pred)

print('the score obtained is:', complement_nb_score)
#14. K nearest neighbors:

from sklearn.neighbors import KNeighborsClassifier

k_neighbors = KNeighborsClassifier(n_neighbors=10)

k_neighbors.fit(X_train, y_train)

k_neighbors_pred = k_neighbors.predict(X_test)

k_neighbors_score = accuracy_score(y_test, k_neighbors_pred)

print('the score obtained is:', k_neighbors_score)
#15. Linear support vector machine classification:

from sklearn.svm import LinearSVC

linear_svc = LinearSVC(loss='squared_hinge', dual=True, tol=0.0001,

                        C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1,

                        class_weight=None, verbose=0, random_state=None, max_iter=1000)

linear_svc.fit(X_train, y_train)

linear_svc_pred = linear_svc.predict(X_test)

linear_svc_score = accuracy_score(y_test, linear_svc_pred)

print('the score obtained is:',linear_svc_score)