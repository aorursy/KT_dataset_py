#This is an attempt to give the easiest take on the titanic problems for absolue beginners who are overwhelmed by the extensive analysis of other solutions

#In no way this is a good analysis, this is so that you get motivated and find other kernels when you are ready to dive in





import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



train = pd.read_csv('../input/train.csv')     #load the train and test data

test = pd.read_csv('../input/test.csv')



#add a new column to maintain the source, we will combine both test and train

train['source']= 'train'

test['source'] = 'test'



titanic = pd.concat([train,test],ignore_index= True)



#fill NA values with median

titanic['Age']=titanic['Age'].fillna(titanic.Age.median())



#titanic.embarked.value_counts gives the value counts of different values, and .index[0] selects the element with index 0

#ie first one ie the highest frequency element

titanic.Embarked = titanic.Embarked.fillna(titanic.Embarked.value_counts().index[0])



titanic["Sex_"] = np.where(titanic.Sex =="male",1,0)

titanic["Embarked_"] = np.where(titanic.Embarked =="C",1,np.where(titanic.Embarked =="Q",2,3))



#Divide back into original

train = titanic.loc[titanic['source'] == "train"]

test = titanic.loc[titanic['source'] == "test"]



from sklearn.linear_model import LogisticRegression

#I am using logistic regression, try with others like RandomForest 

#By doing a few comparisons I decided the age,sex and pclass has the most significance, ignoring plots for easier read

feature_cols = ["Age","Sex_","Pclass"]



X_train = train[feature_cols]

Y_train = train.Survived

X_test = test[feature_cols]



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_log_pred = logreg.predict(X_test)

Y_log_pred_int = Y_log_pred.astype(int)





submission = pd.DataFrame({ "PassengerId": test["PassengerId"], "Survived": Y_log_pred_int })

submission.to_csv('submission4.csv', index=False)