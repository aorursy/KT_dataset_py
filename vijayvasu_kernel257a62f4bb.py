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
import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
train_df=pd.read_csv('/kaggle/input/titanic/train.csv')

test_df=pd.read_csv('/kaggle/input/titanic/test.csv')

dataset_titanic=pd.concat([train_df, test_df])

shiptrain=train_df.copy()

shiptest=test_df.copy()

print(shiptrain.shape)

print(shiptest.shape)





Em_max=train_df.Embarked.value_counts()

print(Em_max)
shiptrain.Embarked.fillna(value=Em_max[0], inplace=True)

shiptest.Embarked.fillna(value=Em_max[0], inplace=True)
shiptrain["Family"]=shiptrain.apply(lambda row: row["Parch"] + row["SibSp"], axis=1)

shiptest["Family"]=shiptest.apply(lambda row: row["Parch"] + row["SibSp"], axis=1)



shiptrain['Gender'] = np.where(shiptrain['Sex']=='male', 1, 0)

shiptest['Gender'] = np.where(shiptest['Sex']=='male', 1, 0)



shiptrain["PortEmbarked"]=shiptrain["Embarked"].map( {'S': 0, 'C': 1, 'Q': 2} )

shiptest["PortEmbarked"]=shiptest["Embarked"].map( {'S': 0, 'C': 1, 'Q': 2} )

shiptest.tail()




shiptrain.Age.fillna(value=shiptrain["Age"].mean(), inplace=True)

shiptest.Age.fillna(value=shiptest["Age"].mean(), inplace=True)

#newship.describe()

shiptrain['AgeBand'] = (pd.cut(shiptrain['Age'], 5))

shiptest['AgeBand'] = (pd.cut(shiptest['Age'], 5))

shiptrain.loc[ shiptrain['Age'] <= 16, 'Age'] = 0

shiptrain.loc[(shiptrain['Age'] > 16) & (shiptrain['Age'] <= 32), 'Age'] = 1

shiptrain.loc[(shiptrain['Age'] > 32) & (shiptrain['Age'] <= 48), 'Age'] = 2

shiptrain.loc[(shiptrain['Age'] > 48) & (shiptrain['Age'] <= 64), 'Age'] = 3

shiptrain.loc[ shiptrain['Age'] > 64, 'Age'] = 4

shiptrain.tail()
shiptest.loc[ shiptest['Age'] <= 16, 'Age'] = 0

shiptest.loc[(shiptest['Age'] > 16) & (shiptest['Age'] <= 32), 'Age'] = 1

shiptest.loc[(shiptest['Age'] > 32) & (shiptest['Age'] <= 48), 'Age'] = 2

shiptest.loc[(shiptest['Age'] > 48) & (shiptest['Age'] <= 64), 'Age'] = 3

shiptest.loc[ shiptest['Age'] > 64, 'Age'] = 4

shiptest.tail()
shiptrain.drop(["AgeBand"], axis=1, inplace=True)

shiptest.drop(["AgeBand"], axis=1, inplace=True)

shiptrain['FareBand'] = (pd.cut(shiptrain['Fare'], 5))



shiptest['FareBand'] = (pd.cut(shiptest['Fare'], 5))
shiptrain.loc[ shiptrain['Fare'] <= 100, 'Fare'] = 1

shiptrain.loc[(shiptrain['Fare'] > 101) & (shiptrain['Fare'] <= 200), 'Fare'] = 2

shiptrain.loc[(shiptrain['Fare'] > 201) & (shiptrain['Fare'] <=300), 'Fare'] = 3

shiptrain.loc[(shiptrain['Fare'] > 301) & (shiptrain['Fare'] <= 400), 'Fare'] = 4

shiptrain.loc[ shiptrain['Fare'] > 400, 'Fare'] = 5
shiptest.loc[ shiptest['Fare'] <= 100, 'Fare'] = 1

shiptest.loc[(shiptest['Fare'] > 101) & (shiptest['Fare'] <= 200), 'Fare'] = 2

shiptest.loc[(shiptest['Fare'] > 201) & (shiptest['Fare'] <=300), 'Fare'] = 3

shiptest.loc[(shiptest['Fare'] > 301) & (shiptest['Fare'] <= 400), 'Fare'] = 4

shiptest.loc[ shiptest['Fare'] > 400, 'Fare'] = 5
shiptrain.drop(["Embarked"],axis=1, inplace=True)

shiptest.drop(["Embarked"],axis=1, inplace=True)

shiptrain.head()
gender_grpby=shiptrain[["Gender", "Survived"]].groupby(["Gender"],as_index=False).mean().sort_values(by="Survived", ascending=False)

gender_grpby
pclass_grpby=shiptrain[["Pclass", "Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived", ascending=False)

pclass_grpby
family_grpby=shiptrain[["Family", "Survived"]].groupby(["Family"],as_index=False).mean().sort_values(by="Survived", ascending=False)

family_grpby

plt.bar(family_grpby.Family, family_grpby.Survived, color="green", edgecolor="yellow")

plt.xlabel("Family Size")

plt.ylabel("Survival mean")

plt.show()
port_grpby=shiptrain[["PortEmbarked", "Survived"]].groupby(["PortEmbarked"],as_index=False).mean().sort_values(by="Survived", ascending=False)

port_grpby
correlation=shiptrain.corr()

sns.heatmap(correlation, vmin=0, vmax=1, annot=True, linewidths=0.5, cmap="Paired")


shiptrain.PortEmbarked.fillna(value=0, inplace=True)


shiptest.Fare.fillna(value=1.0, inplace=True)

#shiptest.Fare.value_counts()
shiptest.isnull().sum()


shiptrain.drop(["Name", "Sex", "SibSp", "Parch", "Ticket", "Cabin"], axis=1, inplace=True)

shiptrain.head()
shiptest.drop(["Name", "Sex", "SibSp", "Parch", "Ticket", "Cabin"], axis=1, inplace=True)

shiptest.head()
shiptrain.drop(["FareBand"], axis=1, inplace=True)
shiptest.drop(["FareBand"], axis=1, inplace=True)
shiptrain.head()
shiptest.head(2)
xvalues= shiptrain[["PassengerId","Pclass", "Age","Fare", "Family", "Gender", "PortEmbarked"]]




yvalue=shiptrain[["Survived"]]





xvalues.head()
yvalue.head(2)
shiptest.head()
#Logictic Regression

Lclass=LogisticRegression()

Lclass.fit(xvalues,yvalue)

LogisticRegressionPrediction=Lclass.predict(shiptest)











#print(LogisticRegressionPrediction)

Logscore=round(Lclass.score(xvalues,yvalue)*100,2)

print(Logscore)
#Support Vector Machine

SVCclass=SVC()

SVCclass.fit(xvalues, yvalue)

#SVCpred=SVCclass.predict(shiptest)

SVCscore=round(SVCclass.score(xvalues, yvalue)*100,2 )

print(SVCscore)





#KNN

Kclass=KNeighborsClassifier(n_neighbors = 3)

Kclass.fit(xvalues, yvalue)

#KNNpred=Kclass.predict(x_test)

KNNscore=round(Kclass.score(xvalues, yvalue)*100,2 )

print(KNNscore)



#NaiveBayes

NBclass=GaussianNB()

NBclass.fit(xvalues, yvalue)



NBscore=round(NBclass.score(xvalues, yvalue)*100,2 )

print(NBscore)



#Decision Trees

DTclass=DecisionTreeClassifier()

DTclass.fit(xvalues, yvalue)



                       

DTscore=round(DTclass.score(xvalues, yvalue)*100,2 )

print(DTscore)



prediction=DTclass.predict(shiptest)

print(prediction)
#Random Forest

RFclass=RandomForestClassifier(n_estimators=100)

RFclass.fit(xvalues, yvalue)

RFpred=RFclass.predict(shiptest)

RFscore=round(RFclass.score(xvalues, yvalue)*100,2 )

print(RFscore)



print(RFpred)
model=pd.DataFrame({

    "Model Name":["Logistic Regression", "SVC", "KNN", "Naive Bayes", "Decision Trees", "Random Forest"],

    "Score":[Logscore,SVCscore,KNNscore,NBscore,DTscore,RFscore]

})

model.sort_values(by="Score", ascending=False).reset_index(drop=True)
shiptest.PassengerId
submission = pd.DataFrame({

        "PassengerId": shiptest.PassengerId,

        "Survived":RFpred

    })

submission.head()
submission.to_csv('prediction.csv', index=False)