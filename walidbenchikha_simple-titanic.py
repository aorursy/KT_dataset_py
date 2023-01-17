# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



#bring data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

survived = df_train.Survived

df_train.head()
# Displaying Scatter Plot

data = pd.concat([df_train['Survived'], df_train['Age']], axis=1)

data.plot.scatter(x='Age', y='Survived');
# BarPlot Display Survival rate by Embarked and Sex

ax = sns.barplot(x="Embarked", y="Survived", hue='Sex', data=df_train)
# BarPlot Display Survival rate by Sex 

ax = sns.barplot(x="Sex", y="Survived", data=df_train)
# BarPlot Display Survival rate by Pclass 

ax = sns.barplot(x="Pclass", y="Survived", data=df_train)
# BarPlot Display Survival rate by Embarked 

ax = sns.barplot(x="Embarked", y="Survived", data=df_train)
# BarPlot Display Survival rate by Embarked 

g = sns.FacetGrid(df_train, col='Survived')

g.map(plt.hist, 'Age', bins=15)
# Display Data Info

print("Train Data Info")

df_train.info()

print()

print("Test Data Null Fields")

number1 = df_test.isna().sum()

print(number1)

def simplify_embarked(dataset):

    Embarked  = pd.get_dummies(  dataset.Embarked , prefix='Embarked'  )

    dataset = dataset.drop(['Embarked'], axis=1)

    dataset= pd.concat([dataset, Embarked], axis=1)  

    # we should drop one of the columns

    dataset = dataset.drop(['Embarked_S'], axis=1)

    return dataset



df_train["Embarked"] = df_train["Embarked"].fillna("S")

df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())

df_train = simplify_embarked(df_train)

df_test = simplify_embarked(df_test)

df_train.head()

labelEncoder_X = LabelEncoder()

df_train.Sex=labelEncoder_X.fit_transform(df_train.Sex)

df_test.Sex=labelEncoder_X.fit_transform(df_test.Sex)
df_train = df_train.drop(['Cabin','Ticket', 'Parch', 'SibSp', 'Survived'], axis=1)

df_test = df_test.drop(['Cabin','Ticket', 'Parch', 'SibSp'], axis=1)
def fill_missing_age(dataset):

    got= dataset.Name.str.split(',').str[1]

    dataset.iloc[:,2]=pd.DataFrame(got).Name.str.split('\s+').str[1]



    title_mean_age=[]

    title_mean_age.append(list(set(dataset.Name)))  #set for unique values of the title, and transform into list

    title_mean_age.append(dataset.groupby('Name').Age.mean())

    title_mean_age



    #------------------ Fill the missing Ages ---------------------------

    n_traning= dataset.shape[0]   #number of rows

    n_titles= len(title_mean_age[1])

    for i in range(0, n_traning):

        if np.isnan(dataset.Age[i])==True:

            for j in range(0, n_titles):

                if dataset.Name[i] == title_mean_age[0][j]:

                    dataset.Age[i] = title_mean_age[1][j]



        if dataset.Age[i] < 18:

            dataset.Age[i]= 0

        elif dataset.Age[i] < 50:

            dataset.Age[i]= 1

        else:

            dataset.Age[i]= 2

    return dataset



df_train = fill_missing_age(df_train)

df_test = fill_missing_age(df_test)



df_train.head(10)

df_train = df_train.drop(['Name'], axis=1)

df_test = df_test.drop(['Name'], axis=1)

df_train.head()
FIELDS = ['Pclass','Sex','Age','Fare','Embarked_C','Embarked_Q']



from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=df_train[FIELDS] , y=survived , cv = 10)

print("Random Forest:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std())
classifier.fit(df_train[FIELDS], survived)

predictions = classifier.predict(df_test[FIELDS])

predictions
# OUTPUT FILE -----------------------------------------------------------------------------------

PassengerId =np.array(df_test["PassengerId"]).astype(int)

my_prediction = pd.DataFrame(predictions, PassengerId, columns = ["Survived"])



my_prediction.to_csv("my_prediction.csv", index_label = ["PassengerId"])



print("The end ...")
