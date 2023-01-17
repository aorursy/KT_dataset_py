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
#import libraries

import seaborn as sns

import matplotlib.pyplot as plt

#for jupyter notebook we use this line

%matplotlib inline    

sns.set_style('whitegrid')
import pandas as pd

#gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

#test = pd.read_csv("../input/titanic/test.csv")

Titanicdata = pd.read_csv("../input/titanic/train.csv")
#Check the 10 five samples for data

Titanicdata.head(10)
#Check the last 10 samples for data

Titanicdata.tail(10)
#check simple information like  columns names ,  columns datatypes and null values

Titanicdata.info()
#check summary of numerical data  such as count , mean , max , min  and standard deviation.

Titanicdata.describe()
#check numbers of rows(samples) and columns(features)

Titanicdata.shape
#check count of values for each features

Titanicdata.count()
#Check total missing values in each feature

Titanicdata.isnull().sum()
#delete ticket, cabin, and passengerID 

Titanicdata.drop(['Ticket','Cabin',"PassengerId"],axis=1,inplace=True)
Titanicdata["Sex"].value_counts()
groubBySurvived=Titanicdata.groupby("Survived").size()

no_Survivors=groubBySurvived[1]

no_Deaths=groubBySurvived[0]

print("Numbers of People Survivers: {} \nNumbers of People Deaths: {}".format(no_Survivors,no_Deaths))
class_sex_grouping = Titanicdata.groupby(['Pclass','Sex']).count()

class_sex_grouping
class_sex_grouping['Survived'].plot.pie()
Embarked_sex_grouping = Titanicdata.groupby(['Embarked','Sex',]).count()

Embarked_sex_grouping
Embarked_sex_grouping['Pclass'].plot.bar()
sns.pairplot(Titanicdata)
sns.countplot(x="Sex",data=Titanicdata)
sns.barplot('Embarked', 'Survived', data=Titanicdata)
sns.barplot('Pclass', 'Survived', data=Titanicdata)