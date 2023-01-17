import pandas as pd

import matplotlib.pyplot as plt
dataset=pd.read_csv('../input/HR_comma_sep.csv')
dataset.head()
#Analysis on Why people left the company
#ploting crosstab for comparing salary and left 

pd.crosstab(dataset.salary,dataset.left).plot(kind='bar')
#ploting crosstab for comparing Department and left

pd.crosstab(dataset.Department,dataset.left).plot(kind='bar')
#so in the plotting we see each department have higher non-left bar which make it non-compareable variable
#removing of department column from dataset

dataset=dataset.drop('Department',axis='columns')
#salary has text data.Need to converted it into float for further analysis

#this is done by labelencoder model from sklearn

#this will add up converted float datatype column into the main dataset frame

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

dataset['salary_n']=labelencoder.fit_transform(dataset['salary'])
dataset.head()
#removing of salary column 

dataset=dataset.drop('salary',axis='columns')
dataset.head()
#intializing the input for model

Y=dataset.left

X=dataset.drop('left',axis='columns')
Y.head()
X.head()


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
#importing logestic regression model 

from sklearn.linear_model import LogisticRegression

model1=LogisticRegression()

model1=model1.fit(X_train,Y_train)
model1.score(X_test,Y_test)
#importing random forest tree 

from sklearn.ensemble import RandomForestClassifier

model2=RandomForestClassifier()

model2=model2.fit(X_train,Y_train)
model2.score(X_test,Y_test)