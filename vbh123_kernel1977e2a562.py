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
# Imports and Helper Functions for 

# data Analysis

import pandas as pd

import numpy as np

import random as rng



# Visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#SciKit Learn Models

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier





from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

data=pd.read_csv("/kaggle/input/ufcdataset/data.csv")

data.info()
data.head()
data.describe()
data.shape
data.describe(include=['O'])
data.tail()


data['B_Age'].fillna(value=int(data['B_Age'].mean()),inplace=True)

data['B_Weight'].fillna(value=int(data['B_Weight'].mean()),inplace=True)

data['B_Height'].fillna(value=int(data['B_Height'].mean()),inplace=True)

data['R_Age'].fillna(value=int(data['R_Age'].mean()),inplace=True)

data['R_Weight'].fillna(value=int(data['R_Weight'].mean()),inplace=True)

data['R_Height'].fillna(value=int(data['R_Height'].mean()),inplace=True)

Extras = []

for i in data.columns:

    counts = data[i].isnull().sum()

    if (counts / len(data)) * 100 > 90:

        Extras.append(i)



len(Extras)

dropdata = data.drop(Extras,axis=1)
dropdata = dropdata.drop(['B_ID','B_Name','R_ID','R_Name','winby','Date'],axis=1)


dropdata.rename(columns={'BPrev':'B__Prev',

                         'RPrev':'R__Prev',

                         'B_Age':'B__Age',

                         'B_Height':'B__Height',

                         'B_Weight':'B__Weight',

                         'R_Age':'R__Age',

                         'R_Height':'R__Height',

                         'R_Weight':'R__Weight',

                         'BStreak':'B__Streak',

                         'RStreak': 'R__Streak'},inplace=True)

dropdata.describe()
dropdata.fillna(value=0,inplace=True)
data.describe(include=['O'])
objecttypes = list(dropdata.select_dtypes(include=['O']).columns)

for col in objecttypes:

    dropdata[col] = dropdata[col].astype('category')

objecttypes
cat_columns = dropdata.select_dtypes(['category']).columns

dropdata[cat_columns] = dropdata[cat_columns].apply(lambda x: x.cat.codes)

dropdata.info()

dropdata.tail()
len(data.columns)
k = 10 #number of variables for heatmap

corrmat = dropdata.corr()

cols = corrmat.nlargest(k, 'winner')['winner'].index

cm = np.corrcoef(dropdata[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

Extras
 #We Store prediction of each model in our dict

# Helper Functions for our models. 

import pickle



def percep(X_train,Y_train,X_test,Y_test,Models):

    perceptron = Perceptron(max_iter = 1000, tol = 0.001)

    perceptron.fit(X_train, Y_train)

    Y_pred = perceptron.predict(X_test)

    Models['Perceptron'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]

    

    Pkl_Filename = "Perceptron.pkl"  

    with open(Pkl_Filename, 'wb') as file:  

        pickle.dump(perceptron, file)

    return



def ranfor(X_train,Y_train,X_test,Y_test,Models):

    randomfor = RandomForestClassifier(max_features="sqrt",

                                       n_estimators = 700,

                                       max_depth = None,

                                       n_jobs=-1

                                      )

    randomfor.fit(X_train,Y_train)

    Y_pred = randomfor.predict(X_test)

    Models['Random Forests'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]

    

    Pkl_Filename = "RandomForest.pkl"  

    with open(Pkl_Filename, 'wb') as file:  

        pickle.dump(randomfor, file)

    return





def dec_tree(X_train,Y_train,X_test,Y_test,Models):

    decision_tree = DecisionTreeClassifier()

    decision_tree.fit(X_train, Y_train)

    Y_pred = decision_tree.predict(X_test)

    Models['Decision Tree'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]

    

    Pkl_Filename = "DecisionTree.pkl"  

    with open(Pkl_Filename, 'wb') as file:  

        pickle.dump(decision_tree, file)

    return





def linSVC(X_train,Y_train,X_test,Y_test,Models):

    linear_svc = LinearSVC()

    linear_svc.fit(X_train, Y_train)

    Y_pred = linear_svc.predict(X_test)

    Models['SVM'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]

    

    Pkl_Filename = "SVM.pkl"  

    with open(Pkl_Filename, 'wb') as file:  

        pickle.dump(linear_svc, file)

    return





def Nearest(X_train,Y_train,X_test,Y_test,Models):

    knn = KNeighborsClassifier(n_neighbors = 3)

    knn.fit(X_train, Y_train)

    Y_pred = knn.predict(X_test)

    Models['KNN'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]

    

    Pkl_Filename = "KNN.pkl"  

    with open(Pkl_Filename, 'wb') as file:  

        pickle.dump(knn, file)





def run_all_and_Plot(df):

    Models = dict()

    from sklearn.model_selection import train_test_split

    X_all = df.drop(['winner'], axis=1)

    y_all = df['winner']

    X_train, X_test, Y_train, Y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)

    percep(X_train,Y_train,X_test,Y_test,Models)

    ranfor(X_train,Y_train,X_test,Y_test,Models)

    dec_tree(X_train,Y_train,X_test,Y_test,Models)

    linSVC(X_train,Y_train,X_test,Y_test,Models)

    Nearest(X_train,Y_train,X_test,Y_test,Models)

    return Models





def plot_bar(dict):

    labels = tuple(dict.keys())

    y_pos = np.arange(len(labels))

    values = [dict[n][0] for n in dict]

    plt.bar(y_pos, values, align='center', alpha=0.5)

    plt.xticks(y_pos, labels,rotation='vertical')

    plt.ylabel('accuracy')

    plt.title('Accuracy of different models')

    plt.show()


accuracies = run_all_and_Plot(dropdata)

CompareAll = dict()

CompareAll['Baseline'] = accuracies

for key,val in accuracies.items():

    print(str(key) +' '+ str(val[0]))

plot_bar(accuracies)
