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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
import numpy as np 

import pandas as pd



# create testing and training sets for hold-out verification using scikit learn method

# test classification dataset

from sklearn.datasets import make_classification

# evaluate a logistic regression model using k-fold cross-validation

from numpy import mean

from numpy import std

from sklearn.datasets import make_classification

from sklearn.model_selection import KFold

from sklearn.model_selection import RepeatedKFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error



#import seaborn and plotly

import matplotlib

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

import re



from IPython.display import Image as PImage

from PIL import Image, ImageDraw, ImageFont



#ignore warning

import warnings

warnings.filterwarnings('ignore')
train=train_data.copy()

test=test_data.copy()
from matplotlib.pyplot import figure

figure(figsize=(10, 4))



survived = train[train["Survived"] == 1]

died = train[train["Survived"] == 0]

survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)

died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)

plt.legend(['Survived','Died'])

plt.show()
# https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset

# Copy original dataset in case we need it later when digging into interesting features

# WARNING: Beware of actually copying the dataframe instead of just referencing it

# "original_train = train" will create a reference to the train variable (changes in 'train' will apply to 'original_train')

original_train = train.copy() # Using 'copy()' allows to clone the dataset, creating a different object with the same values



# Feature engineering steps taken from Sina and Anisotropic, with minor changes to avoid warnings

full_data = [train, test]



# Feature that tells whether a passenger had a cabin on the Titanic

train['HasCabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['HasCabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLS in the Fare column

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())



# Remove all NULLS in the Age column

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    # Next line has been improved to avoid warning

    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)



# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] ;
for dataset in full_data:

    # feature extraction: has ticket string

    dataset['HasTicket'] = dataset['Ticket'].apply(lambda x: 1 if x else 0).astype('object')
# Feature selection: remove variables no longer containing relevant information

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

test = test.drop(drop_elements, axis = 1)
print(train.info())
#check cleaning data

train.head(70)
#check cleaning data

test.head(70)
#define train and test set



Xtrain = train.copy()

Xtrain = Xtrain.drop(['Survived'], axis=1)

Ytrain = train[['Survived']].astype(int)



# test.csv does not have any Survived field

#Xtest = test[pd.isnull(test['Survived'])].drop(['Survived'], axis=1)

#Ytest = test[['Survived']].astype(int)

print(Xtrain)
print(Ytrain)
#see https://scikit-learn.org/stable/modules/model_evaluation.html



modelist=[

    #['Linear',LinearRegression()],

    ['Decision Tree classifier',DecisionTreeClassifier()],

    ['Gaussian Naive Bayes ',GaussianNB()],

    ['Neural Network Multi-layer Perceptron classifier ',MLPClassifier(hidden_layer_sizes=1, activation='relu', solver='adam',learning_rate='constant',max_iter=200)]

]



from tabulate import tabulate

pdtabulate=lambda df:tabulate(df,headers=['Model','Precision','Recall','fMeasure'],tablefmt='pretty', showindex='always')

datalist=[]



pc=[]

rc=[]

fm=[]



for name,model in (modelist):

  precision = cross_val_score(model, Xtrain,Ytrain, cv=2, scoring ='precision_weighted')

  recall = cross_val_score(model, Xtrain,Ytrain, cv=2, scoring ='recall_weighted')

  fmeasure = cross_val_score(model, Xtrain,Ytrain, cv=2, scoring ='f1_weighted')

  #print(' %s precision %f recall %f fmeasure %f  avg %f'%(name,np.mean(precision),np.mean(recall),np.mean(fmeasure)))

  datalist.append( [name,np.mean(precision),np.mean(recall),np.mean(fmeasure)])

  pc.append(np.mean(precision))

  rc.append(np.mean(recall))

  fm.append(np.mean(fmeasure))

    



datalist.append( ['Average all',np.mean(pc),np.mean(rc),np.mean(fm)])

print(pdtabulate(datalist))  
model=DecisionTreeClassifier(max_depth=4)

model.fit(Xtrain,Ytrain)



 



import graphviz 



dot_data = tree.export_graphviz(model, 

                                out_file=None,

                                filled=True, 

                                rounded=True,  

                                special_characters=True) 

graph = graphviz.Source(dot_data)

graph
