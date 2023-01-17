import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler

    
#Visualize data lize data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.info();

print("----------------------------")

test.info();
#detailed description 

train.describe(include="all")

print("----------------------------")

test.describe(include="all")

train.sample(5)
test.sample(5)
#as cabin is missing more than 50% of its data it will be dropped

# Name and Ticket.nO wont be useful so it will also be dropped



train=train.drop(['Cabin','Ticket','Name'],axis=1)

test=test.drop(['Cabin', 'Ticket','Name'],axis=1)
#Column sex has 2 Catagories  ---> male and female 

#Column Embarked has 3 Catagories ----> Q S C

#Column Pclass has 3 Catagories ----> 1 2 3

#Column Parch has 6 Catagories ----> 1 2 3 4 5  6

#visualising columns Sex and Catagories
#bar plot of Sex vs  Survival

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))

sns.barplot(x="Sex", y="Survived", data=train, ax=axis2)

sns.countplot(x='Sex', data=train, ax=axis1)

print("Percentage of People who Survived")



female = train["Survived"][train["Sex"] == 'female' ].value_counts(normalize = True)

female_count = train["Survived"][train["Sex"] == 'female' ].value_counts()

male = train["Survived"][train["Sex"] == 'male' ].value_counts(normalize = True)

male_count = train["Survived"][train["Sex"] == 'male' ].value_counts()





print("Female :",(female[1] * 100) , "    Male: " , (male[1] * 100) )

print("Total Count:")

print("Female :",(female_count[1] + female_count[0]) , "    Male: " , (male_count[1] + male_count[0]))
#bar plot of Embarked vs Survived

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))

sns.barplot(x="Embarked", y="Survived", data=train , ax=axis2)

sns.countplot(x='Embarked', data=train, ax=axis1)



S = train["Survived"][train["Embarked"] == 'S' ].value_counts(normalize = True)

S_count = train["Survived"][train["Embarked"] == 'S' ].value_counts()

C = train["Survived"][train["Embarked"] == 'C' ].value_counts(normalize = True)

C_count = train["Survived"][train["Embarked"] == 'C' ].value_counts()

Q = train["Survived"][train["Embarked"] == 'Q' ].value_counts(normalize = True)

Q_count = train["Survived"][train["Embarked"] == 'Q' ].value_counts()



print("Percentage of People who Survived")

print("S :",(S[1] * 100) , "    C: " , (C[1] * 100) ,  "    Q: " , (Q[1] * 100) )

print("Total Count:")

print("S :",(S_count[1] + S_count[0]) , "    C: " , (C_count[1] + C_count[0]) ,  "    Q: " , (Q_count[1] +Q_count[0] ))
fig, (axis1 , axis2) = plt.subplots(1,2,figsize = (15,5))



sns.barplot(x="Pclass", y="Survived", data=train , ax=axis2)

sns.countplot(x='Pclass', data=train, ax=axis1)



one = train["Survived"][train["Pclass"] == 1 ].value_counts(normalize = True)

one_count = train["Survived"][train["Pclass"] == 1 ].value_counts()

two = train["Survived"][train["Pclass"] == 2 ].value_counts(normalize = True)

two_count = train["Survived"][train["Pclass"] == 2 ].value_counts()

three = train["Survived"][train["Pclass"] == 3 ].value_counts(normalize = True)

three_count = train["Survived"][train["Pclass"] == 3 ].value_counts()



print("Percentage of People who Survived")

print("1 :",(one[1] * 100) , "    2: " , (two[1] * 100) ,  "    3: " , (three[1] * 100) )

print("Total Count:")

print("1 :",(one_count[1] + one_count[0]) , "    2: " , (two_count[1] + two_count[0]) ,  "    3: " , (three_count[1] +three_count[0] ) )
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))

sns.countplot(x='Parch', data=train , ax=axis1)

sns.barplot(x="Parch", y="Survived", data=train , ax=axis2)



#this column defines the number of people 
#Splitting and joining Embarked column

train["Embarked"] = train["Embarked"].fillna("S")

train_dummies = pd.get_dummies(train["Embarked"])

test_dummies = pd.get_dummies(test["Embarked"])

train = train.join(train_dummies)

test = test.join(test_dummies)
#Splitting and joining Sex column

train_dummies = pd.get_dummies(train["Sex"])

test_dummies = pd.get_dummies(test["Sex"])

train = train.join(train_dummies)

test = test.join(test_dummies)
#Splitting and joining Pclass column

train_dummies = pd.get_dummies(train["Pclass"])

test_dummies = pd.get_dummies(test["Pclass"])

train = train.join(train_dummies)

test = test.join(test_dummies)
fig, (axis1, axis2) = plt.subplots(1,2,figsize=(15,5))

sns.countplot(x="Age",data=train,ax=axis1)

sns.barplot(x="Age",y="Survived",ax=axis2,data=train)
train = train.drop(["Embarked","Sex","Pclass"] , axis=1)

test = test.drop(["Embarked","Sex","Pclass"] , axis=1)
train["Age"] = train["Age"].fillna('0')

test["Age"] = test["Age"].fillna('0')

test.describe(include="all")
test.sample(10)

test["Fare"] = test["Fare"].fillna('26.0000')
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler



y_train = train["Survived"]

x_train = train.drop(["Survived","PassengerId"],axis=1)



sc_x = StandardScaler()

scaler = StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)



Xtrain, xtest, ytrain, ytest = train_test_split(x_train,y_train,test_size=0.2,random_state=0)



#random forestt

random_forest = RandomForestClassifier(n_estimators = 500)

random_forest.fit(Xtrain,ytrain)

y_predict_forest = random_forest.predict(xtest)

acc_forest = round(accuracy_score(y_predict_forest,ytest)*100,2)

print(acc_forest)
#gaussian 

gaussian = GaussianNB()

gaussian.fit(Xtrain,ytrain)

y_predict_gaussian = gaussian.predict(xtest)

gaussian.score(x_train,y_train)

acc_gaussian = round(accuracy_score(y_predict_gaussian,ytest)*100,2)

print(acc_gaussian)
#KNneighbour

knn = KNeighborsClassifier(n_neighbors = 3 , p=1)

knn.fit(Xtrain,ytrain)

y_predict_KN = knn.predict(xtest)

acc_KN = round(accuracy_score(y_predict_KN,ytest)*100,2)

print(acc_KN)
from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(Xtrain,ytrain)

y_predict_SGD = sgd.predict(xtest)

acc_SGD = round(accuracy_score(y_predict_SGD,ytest)*100,2)

print(acc_SGD)
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(Xtrain,ytrain)

y_predict_boosting = gbk.predict(xtest)

acc_boosting = round(accuracy_score(y_predict_boosting,ytest)*100,2)

print(acc_boosting)
from sklearn.tree import DecisionTreeClassifier



DecisionTree = DecisionTreeClassifier()

DecisionTree.fit(Xtrain,ytrain)

y_predict_DecisionTree = DecisionTree.predict(xtest)

acc_DecisionTree = round(accuracy_score(y_predict_DecisionTree,ytest)*100,2)

print(acc_DecisionTree)
from sklearn.linear_model import Perceptron



prep = Perceptron()

prep.fit(Xtrain,ytrain)

y_predict_prep = prep.predict(xtest)

acc_prep = round(accuracy_score(y_predict_prep,ytest)*100,2)

print(acc_prep)
from sklearn.svm import SVC



svc = SVC()

svc.fit(Xtrain,ytrain)

y_predict_svc = svc.predict(xtest)

acc_svc = round(accuracy_score(y_predict_svc,ytest)*100,2)

print(acc_svc)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(Xtrain,ytrain)

y_predict_logreg = logreg.predict(xtest)

acc_logreg = round(accuracy_score(y_predict_logreg,ytest)*100,2)

print(acc_logreg)
from sklearn.svm import LinearSVC



linersvc = LinearSVC()

linersvc.fit(Xtrain,ytrain)

y_predict_linersvc = linersvc.predict(xtest)

acc_linersvc = round(accuracy_score(y_predict_linersvc,ytest)*100,2)

print(acc_linersvc)
models = pd.DataFrame({

    'Model':["SVC","KNN","LogisR","RandomForest","Perceptron","LinerSVC","DecissionTree","SGD","GradientBoosting"]

    ,

    "Score":[acc_svc,acc_KN,acc_logreg,acc_forest,acc_prep,acc_linersvc,acc_DecisionTree,acc_SGD,acc_boosting]

})



models.sort_values(by="Score",ascending=False)



print(Xtrain.shape)
from keras.models import Sequential



# Import `Dense` from `keras.layers`

from keras.layers import Dense



# Initialize the constructor

model = Sequential()



# Add an input layer 

model.add(Dense(20, activation='relu', input_shape=(12,)))



# Add one hidden layer 

model.add(Dense(12, activation='relu'))



model.add(Dense(12, activation='relu'))

# Add an output layer 

model.add(Dense(12, activation='relu'))



model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.fit(Xtrain,ytrain,epochs=5, batch_size=1, verbose=1)

score = model.evaluate(xtest, ytest,verbose=1)

print(score)

y_train = test.drop("PassengerId",axis=1)

ids = test["PassengerId"]

scaler1 = StandardScaler().fit(y_train)

y_train = scaler1.transform(y_train)

y_pred = model.predict(y_train)

y_pred = np.round(y_pred)

y_pred = np.reshape(y_pred,418)
Predictions = gbk.predict(sc_x.fit_transform(test.drop("PassengerId",axis=1)))

output = pd.DataFrame({"PassengerId":ids , "Survived": y_pred})

output.to_csv("submission.csv",index=False)