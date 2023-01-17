# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn import svm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.metrics import classification_report

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import random

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

# Any results you write to the current directory are saved as output.
# Data cleanup

# TRAIN DATA

train_df = pd.read_csv('../input/train.csv', header=0)        # Load the train file into a dataframe

# TEST DATA

test_df = pd.read_csv('../input/test.csv', header=0)        # Load the test file into a dataframe

ID = test_df["PassengerId"]

# I need to convert all strings to integer classifiers.

# I need to fill in the missing values of the data and make it complete.



# female = 0, Male = 1

#train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

train_df['Sex']=train_df['Sex'].astype('category')

train_df['Sex_cat']=train_df['Sex'].cat.codes



test_df['Sex']=test_df['Sex'].astype('category')

test_df['Sex_cat']=test_df['Sex'].cat.codes



# Embarked from 'C', 'Q', 'S'

# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.



mode_emb_train = train_df['Embarked'].mode()

train_df['Embarked']=train_df['Embarked'].fillna(mode_emb_train)

train_df['Embarked']=train_df['Embarked'].astype('category')

train_df['Embarked_cat']=train_df['Embarked'].cat.codes

###############

mode_emb_test = test_df['Embarked'].mode()

test_df['Embarked']=test_df['Embarked'].fillna(mode_emb_test)

test_df['Embarked']=test_df['Embarked'].astype('category')

test_df['Embarked_cat']=test_df['Embarked'].cat.codes



# Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,

# Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index

# train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

train_df['Embarked']=train_df['Embarked'].astype('category')

train_df['Embarked_cat']=train_df['Embarked'].cat.codes



# #Fare

# train_df['Fare'].replace('0',None,inplace=True)

# mean_fare = train_df['Fare'].dropna().mean()

# if len(train_df.Fare[ train_df.Fare.isnull() ]) > 0:

#     train_df.loc[ (train_df.Fare.isnull()), 'Fare'] =  mean_fare

# #Alternative method (common practice)       



# ave_age_train = train_df['Age'].dropna().mean()

# std_age_train = train_df['Age'].dropna().std()

# ave_age_test = test_df['Age'].dropna().mean()

# std_age_test = test_df['Age'].dropna().std()

        

# random.seed(42)

# train_df['Age']=train_df['Age'].fillna(ave_age_train + random.uniform(-1,1) * std_age_train)

# test_df['Age']=test_df['Age'].fillna(ave_age_test + random.uniform(-1,1) * std_age_test)

age_train = np.zeros((2,3))

age_test = np.zeros((2,3))



for i in range(0,2):

    for j in range(0,3):

        age_train[i,j] = train_df['Age'][(train_df['Sex_cat'] == i) & (train_df['Pclass'] == j+1)].mean()

        age_test[i,j] = test_df['Age'][(test_df['Sex_cat'] == i) & (test_df['Pclass'] == j+1)].mean()



for i in range(0,2):

    for j in range(0,3):

        train_df.loc[(train_df['Age'].isnull())&(train_df['Sex_cat'] == i)&(train_df['Pclass'] == j+1),'Age'] = age_train[i,j] 

        test_df.loc[(test_df['Age'].isnull())&(test_df['Sex_cat'] == i)&(test_df['Pclass'] == j+1),'Age'] = age_test[i,j]   



# Ports = list(enumerate(np.unique(test_df['Embarked'])))    # determine all values of Embarked,

# Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index

# test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

test_df['Embarked']=test_df['Embarked'].astype('category')

test_df['Embarked_cat']=test_df['Embarked'].cat.codes



# # Fare

# test_df['Fare'].replace('0',None,inplace=True)

# mean_fare = test_df['Fare'].dropna().mean()

# if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:

#     test_df.loc[ (test_df.Fare.isnull()), 'Fare'] =  mean_fare

train_fare_trans = train_df['Fare'].groupby(train_df['Pclass'])

test_fare_trans = test_df['Fare'].groupby(test_df['Pclass'])



f = lambda x : x.fillna(x.mean())

train_df['Fare'] = train_fare_trans.transform(f)

test_df['Fare'] = test_fare_trans.transform(f)

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)

train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId',"Embarked"], axis=1) 

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)

test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId',"Embarked"], axis=1) 
data = train_df.iloc[:,1:]

target = train_df.Survived

test = test_df

test_y = pd.read_csv("../input/genderclassmodel.csv")["Survived"]
# instantiate a svm model, and fit with X and y

steps = [('scaler', StandardScaler()),('svm', svm.SVC())]

model = Pipeline(steps)

model = model.fit(data, target)

result = model.predict(test)

print(model.score(data,target))

print(model.score(test,test_y))

print(classification_report(test_y, result))
result_df = pd.DataFrame({"PassengerID": ID,"Survived" : result})
result_df.to_csv("titanic.csv",index=False)