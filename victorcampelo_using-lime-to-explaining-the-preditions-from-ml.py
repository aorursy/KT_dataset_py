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
# Imports

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



import numpy as np



import pandas as pd
# Data Loading

df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
df[df['SARS-Cov-2 exam result']=='positive'].head()
df.dtypes.head(10)
df.index
df.columns
df.count()
df = df.drop(columns=['Patient ID'])
df2 = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

df2.corr()
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(100, 100))

sns.heatmap(df.corr(),

            annot = True,

            fmt = '.2f',

            cmap='Blues')

plt.title('Heatmap - Correlation coefficients between variables')

plt.show()
y = pd.get_dummies(df['SARS-Cov-2 exam result'])
#Test Positive covid-19 correlation matrix

plt.figure(figsize=(100, 20))

df.reset_index(drop=True)

k = 10 #number of variables for heatmap

df3 = pd.get_dummies(df.fillna(0))

corrmat = df3.corr()

cols = corrmat.nlargest(k, 'SARS-Cov-2 exam result_positive')

cm = np.corrcoef(df3[cols.index].values.T)

hm = sns.heatmap(cm, cmap='Blues',cbar=True, annot=True, square=True, fmt='.2f', yticklabels=cols.index, xticklabels=cols.index)

plt.show()
#Test Negative covid-19 correlation matrix

plt.figure(figsize=(100, 20))

df.reset_index(drop=True)

k = 10 #number of variables for heatmap

df3 = pd.get_dummies(df.fillna(0))

corrmat = df3.corr()

cols = corrmat.nlargest(k, 'SARS-Cov-2 exam result_negative')

cm = np.corrcoef(df3[cols.index].values.T)

hm = sns.heatmap(cm, cmap='Blues',cbar=True, annot=True, square=True, fmt='.2f', yticklabels=cols.index, xticklabels=cols.index)

plt.show()
full_null_series = (df.isnull().sum() == df.shape[0])

full_null_columns = full_null_series[full_null_series == True].index

# columns with all values equal null

print(full_null_columns.tolist())
df.drop(full_null_columns, axis=1, inplace=True)
df2 = pd.get_dummies(df.fillna(0))

df2
from sklearn.datasets import make_regression, load_boston

from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import RFE, RFECV, f_regression

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.metrics import r2_score

from sklearn.model_selection import ShuffleSplit

from sklearn.preprocessing import StandardScaler, MinMaxScaler

x = df2.loc[:, (df2.columns != 'SARS-Cov-2 exam result_negative') & (df2.columns != 'SARS-Cov-2 exam result_positive') ]

names = df2.columns



#using linear regression as models

lr = LinearRegression()

rfe = RFE(estimator=lr, n_features_to_select=1)

rfe.fit(x, y['positive'])
print("Attributes sorted by rank hair:")

print(sorted(zip(rfe.ranking_, names)))
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import seaborn as sns

X = df2.loc[:, (df2.columns != 'SARS-Cov-2 exam result_negative') & (df2.columns != 'SARS-Cov-2 exam result_positive') ]

Y = y['positive']

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)# training model

model  = RandomForestClassifier()

model.fit(X_train, y_train)# show the importance from each feature

model.feature_importances_
plt.figure(figsize=(15,100))

importances = pd.Series(data=model.feature_importances_, index=X.columns)

sns.barplot(x=importances, y=importances.index, orient='h').set_title('Importances from each feature')
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif, mutual_info_regression
X = df2.loc[:, (df2.columns != 'SARS-Cov-2 exam result_negative') & (df2.columns != 'SARS-Cov-2 exam result_positive') ]

Y = y['positive']

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)

mi = mutual_info_classif(X_train, y_train)
mi = pd.Series(mi)

mi.index = X_train.columns

mi = mi.sort_values(ascending=True)
mi.plot.barh(figsize=(10,50))
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



def models(X_train, Y_train, X_test, Y_test):

    from sklearn.linear_model import LogisticRegression

    log = LogisticRegression(random_state = 0)

    log.fit(X_train, Y_train)

    log_pred = log.predict(X_test)

    

    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

    knn.fit(X_train, Y_train)

    knn_pred = knn.predict(X_test)

    

    from sklearn.svm import SVC

    svc_lin = SVC(kernel='linear', random_state = 0, probability=True)

    svc_lin.fit(X_train, Y_train)

    svc_lin_pred = svc_lin.predict(X_test)

    

    from sklearn.svm import SVC

    svc_rbf = SVC(kernel='rbf', random_state = 0)

    svc_rbf.fit(X_train, Y_train)

    svc_rbf_pred = svc_rbf.predict(X_test)

    

    from sklearn.naive_bayes import GaussianNB

    gauss = GaussianNB()

    gauss.fit(X_train, Y_train)

    gauss_pred = gauss.predict(X_test)

    

    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

    tree.fit (X_train, Y_train)

    tree_pred = tree.predict(X_test)

    

    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state = 0)

    forest.fit(X_train, Y_train)

    forest_pred = forest.predict(X_test)

    

    from sklearn.linear_model import Perceptron

    pcp = Perceptron(random_state = 0)

    pcp.fit(X_train, Y_train)

    pcp_pred = pcp.predict(X_test)

    

    print('[0]Logistic Regression Training Accuracy: ', log.score(X_train, Y_train))

    print('[0]Logistic Regression Testing Accuracy: ', accuracy_score(Y_test, log_pred))

    print(classification_report(Y_test, log_pred))

    

    print('[1]KNeighborns Training Accuracy: ', knn.score(X_train, Y_train))

    print('[1]KNeighborns Testing Accuracy: ', accuracy_score(Y_test, knn_pred))

    print(classification_report(Y_test, knn_pred))

    

    print('[2]SVC Linear Training Accuracy: ', svc_lin.score(X_train, Y_train))

    print('[2]SVC Linear Testing Accuracy: ', accuracy_score(Y_test, svc_lin_pred))

    print(classification_report(Y_test, svc_lin_pred))

    

    print('[3]SVC RBF Training Accuracy: ', svc_rbf.score(X_train, Y_train))

    print('[3]SVC RBF Testing Accuracy: ', accuracy_score(Y_test, svc_rbf_pred))

    print(classification_report(Y_test, svc_rbf_pred))

    

    print('[4]Gaussian NB Training Accuracy: ', gauss.score(X_train, Y_train))

    print('[4]Gaussian NB Testing Accuracy: ', accuracy_score(Y_test, gauss_pred))

    print(classification_report(Y_test, gauss_pred))

    

    print('[5]Decision Tree Training Accuracy: ', tree.score(X_train, Y_train))

    print('[5]Decision Tree Testing Accuracy: ', accuracy_score(Y_test, tree_pred))

    print(classification_report(Y_test, tree_pred))

    

    print('[6]Random Forest Training Accuracy: ', forest.score(X_train, Y_train))

    print('[6]Random Forest Testing Accuracy: ', accuracy_score(Y_test, forest_pred))

    print(classification_report(Y_test, forest_pred))

    

    print('[7] Perceptron Training Accuracy: ', pcp.score(X_train, Y_train))

    print("[7] Perceptron Testing accuracy: ", accuracy_score(Y_test, pcp_pred))

    print(classification_report(Y_test, pcp_pred))

    

    return log, knn, svc_lin, svc_rbf, gauss, tree, forest, pcp
mod = models(X_train,y_train, X_test, y_test)
df[df.index == 3462]
predict_fn = lambda x: mod[0].predict_proba(x).astype(float)

predict_fn1 = lambda x: mod[1].predict_proba(x).astype(float)

predict_fn2 = lambda x: mod[5].predict_proba(x).astype(float)

predict_fn3 = lambda x: mod[6].predict_proba(x).astype(float)

predict_fn4 = lambda x: mod[2].predict_proba(x).astype(float) # SVC
import lime

import lime.lime_tabular



explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,                                            

                 feature_names=X_train.columns.values.tolist(),                                        

                 class_names=y_train.unique())



np.random.seed(42)

exp = explainer.explain_instance(X_test.values[465], predict_fn, num_features = 8)

exp.show_in_notebook(show_all=True) #only the features used in the explanation are displayed



exp = explainer.explain_instance(X_test.values[465], predict_fn1, num_features = 8)

exp.show_in_notebook(show_all=True)



exp = explainer.explain_instance(X_test.values[465], predict_fn2, num_features = 8)

exp.show_in_notebook(show_all=True)



exp = explainer.explain_instance(X_test.values[465], predict_fn3, num_features = 8)

exp.show_in_notebook(show_all=True)



# SVC

exp = explainer.explain_instance(X_test.values[465], predict_fn4, num_features = 8)

exp.show_in_notebook(show_all=True)