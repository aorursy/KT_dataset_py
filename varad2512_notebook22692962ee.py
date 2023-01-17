# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

titanic_data = pd.read_csv("../input/train.csv")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



'''

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

'''

titanic_data = titanic_data.drop(['Name'],axis = 1)

titanic_data.info()

titanic_data.head()

titanic_data = titanic_data.drop(['PassengerId'], axis = 1)

titanic_data.head()

titanic_data['Embarked'] = titanic_data['Embarked'].fillna("S")

titanic_data['Cabin'] = titanic_data['Cabin'].fillna("S")

titanic_data.head()



#sns.factorplot('Survived','Fare',data = titanic_data, kind = 'bar')





titanic_data = titanic_data.drop(['Embarked','Ticket','Cabin'],axis = 1)



'''

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize = (15,5))

sns.countplot(x = 'Embarked', data = titanic_data, ax = axis1)

plt.show()

'''



titanic_data.head()



# Any results you write to the current directory are saved as output.
'''Fare TODO Test data pre processing'''

titanic_data['Fare'] = titanic_data['Fare'].astype("int")

titanic_data.head()
'''Age TODO Test data preprocessing'''

count_nan_age_titanic = titanic_data["Age"].isnull().sum()

std_age  = titanic_data["Age"].std()

mean_age = titanic_data["Age"].mean()

random_created = np.random.randint(mean_age - std_age,mean_age + std_age, size = count_nan_age_titanic)

titanic_data["Age"][np.isnan(titanic_data["Age"])] = random_created

titanic_data["Age"] = titanic_data["Age"].astype("int")

titanic_data.head()
titanic_data["Family"] = titanic_data["Parch"] + titanic_data["SibSp"]

titanic_data["Family"].loc[titanic_data["Family"] > 0]  = 1

titanic_data["Family"].loc[titanic_data["Family"] == 0] = 0

titanic_data
titanic_data["Sex"].loc[titanic_data["Sex"] == 'male'] = 1

titanic_data["Sex"].loc[titanic_data["Sex"] == 'female'] = 0

dum = pd.get_dummies(titanic_data["Pclass"])

dum.columns = ['1','2','3']

titanic_data = titanic_data.join(dum)

titanic_data = titanic_data.drop("Pclass",axis=1,inplace=True)

titanic_data.head()