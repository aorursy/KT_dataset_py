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
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
#reading training and testing data

import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()
train.describe()
train.info()
#count NA values

train.isnull().sum()
#here we are droping passengerId bcoz there is no need for prediction

#we droping Cabin bcoz there is 80% NaN value and ticket also

train=train.drop(["PassengerId","Ticket","Cabin"],axis=1)

train.head()
# filling NaN values using pandas Fillna



train["Age"].fillna(train["Age"].mean(),inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)
#visualisation how many pasanger survived and how many dead 

#here we make a function for visulisation bar chart 



def bar_chart(column):

    survived=train[train["Survived"]==1][column].value_counts()

    dead=train[train["Survived"]==0][column].value_counts()

    df1=pd.DataFrame([survived,dead])

    df1.index=["Survived","Dead"]

    df1.plot(kind="bar",figsize=(10,5))
#here we can see relation between sex and Survived

bar_chart("Sex")
#by this visualisation we can see relation between pasenger class and Survived

bar_chart("Pclass")
# here we make another function to visualisation  boxplot to see the outliers

def box_plot(column):

    train.boxplot(by="Survived",column=[column],grid=True)



#here we drow boxplot between Fare and Survived

box_plot("Fare")
#here we drow boxplot between Age and Survived

box_plot("Age")
# here we use histogram .A histogram displays the shape and spread of continuous sample data.



def hist(column):

    train[column].plot(kind = 'hist',bins = 200,figsize = (6,6))

    plt.title(column)

    plt.xlabel(column)

    plt.ylabel("Frequency")

    plt.show()
#making histogram on Age

hist("Age")
# here we use scater plot to display data  

    

train[train.Survived ==1].plot(kind='scatter', x='Age', y='Fare',color="red")

plt.xlabel("Age")

plt.ylabel("Fare")

plt.title("Survived ==1")

plt.grid(True)

plt.show()
#checking feature dependance on outcome

import seaborn as sns

plt.figure(figsize=(10, 10))

sns.heatmap(train.corr(), annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")

plt.show()
#here we splet data in X(input) and y(outcome) . bcoz every Ml algo take input and output for trained

#we also drop Name feature from input data bcoz Name is not important to prediction

X=train.drop(["Name","Survived"],axis=True)

y=train["Survived"]
#here we use one hot encoading to scaling catagorical features

#we need this bcoz every Ml algo takes only num value

x=pd.get_dummies(X)

x.shape
#here we Use MinMax Scaler to scaling the data in same (0 -1) scaling

from sklearn.preprocessing import MinMaxScaler

scalerX = MinMaxScaler(feature_range=(0, 1))

x[x.columns] = scalerX.fit_transform(x[x.columns])

#here we split traing data in to train and test bcoz we use X_train to trained the model

# y_test to check algo accuracy

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=40)
#1st we use LogisticRegression



from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(random_state=40)

lr.fit(x_train,y_train)



print(lr.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(random_state=40,min_impurity_decrease=0.002,min_weight_fraction_leaf=0.001)



rfc.fit(x_train,y_train)



print(rfc.score(x_test,y_test))
from sklearn.metrics import confusion_matrix



def matrix(predection):

    result=confusion_matrix(y_test,predection)

    print(result)