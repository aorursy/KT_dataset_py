# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

import math

def convert(x):

    str = x.replace(".","").split(",")[1].split()

    for y in str:

        if y in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Mr','Sir']:

            return "Mr"

        elif y in ['Countess', 'Mme','Mrs','Lady']:

            return "Mrs"

        elif y in ['Mlle', 'Ms','Miss']:

            return "Ms"

        elif y in ['Dr']:

            return "Dr"

        elif y in ['Master']:

            return "Master"

    print(str)

    return "NAN"

def convertCabin(x):

    print(x)

    if pd.isnull(x):

        return "NAN"

    if x[:1] in ["A","B","C","D","E","F","G","T"]:

        return x[:1]

    else:

        print(x)

        return "NAN"

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv('../input/train.csv')

data["Decks"] = data["Cabin"].map(lambda x: convertCabin(x))

data["Survived"].astype('category')

data["Title"] = data["Name"].map(lambda x: convert(x))



data["Title"].astype('category')

data["Sex"].astype('category')

data["Decks"].astype('category')

del(data["Ticket"])

del(data["Name"])

del(data["PassengerId"])

del(data["Cabin"])

clf = DecisionTreeClassifier(random_state=0)

y = data["Survived"]

del(data["Survived"])

print(data)

#cross_val_score(clf,data,y) 



data.to_csv('out.txt',sep='\t')



# Any results you write to the current directory are saved as output.







        