#importing library
import pandas as pd
import matplotlib.pyplot as plt
#read the data
advertising = pd.read_csv('../input/Advertising.csv')
#check the data
advertising.head()
#Drop unnecessary columns
drop_elements = ['Unnamed: 0']
advertising = advertising.drop(drop_elements, axis=1)
#check the tail of data
advertising.tail()
#check the missing columns
advertising.isnull().values.any()
#Plotting sales against TV
fig,axes = plt.subplots(1,1,figsize=(8,6),sharey=True)
axes.scatter(advertising.TV.values,advertising.Sales.values)
axes.set_xlabel(" TV")
axes.set_ylabel(" Sales")
#Plotting sales against Radio
fig,axes = plt.subplots(1,1,figsize=(8,6),sharey=True)
axes.scatter(advertising.Radio.values,advertising.Sales.values)
axes.set_xlabel(" Radio")
axes.set_ylabel(" Sales")
#Plotting sales against Newspapers
fig,axes = plt.subplots(1,1,figsize=(8,6),sharey=True)
axes.scatter(advertising.Newspaper.values,advertising.Sales.values)
axes.set_xlabel(" Newspapers")
axes.set_ylabel(" Sales")
#Multiple Linear Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
features = ["TV","Radio","Newspaper"]
x=advertising[features]
y=advertising.Sales
model=lm.fit(x,y)
print (" Model coefficients are: ")
print ( model.coef_)
xpredicted = model.predict(x)
print("R-squared of the model")
print(model.score(x,y))
#cross validation
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
features = ["TV","Radio","Newspaper"]
x=advertising[features]
y=advertising.Sales

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)
print("The length of tranning size is %d" % len(X_train))
print("The length of test size is %d " % len(X_test))
model = lm.fit(X_train,y_train)
print("The R-squared value of the model is %.2f" % model.score(X_test,y_test))

#Testing 10-times the linear regression model for the Advertising data set.
for i in range(0,15):
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
    model = lm.fit(X_train,y_train)
    print(model.score(X_test,y_test))
#K- Fold Cross Validation
from sklearn.model_selection import KFold
import numpy as np

lm = LinearRegression()
features = ["TV","Radio","Newspaper"]
x=advertising[features]
y=advertising.Sales

kf = KFold(n_splits=10)
scores=[]
for train,test in kf.split(x,y):
    model = lm.fit(x.values[train],y.values[train])
    score = model.score(x.values[test],y.values[test])
    print(score)
    scores.append(score)
    
print("The mean score for %d-fold cross validation is %.2f" % (kf.get_n_splits(),np.mean(np.array(scores))))

#cross_val_score method
lm = LinearRegression()
features = ["TV","Radio","Newspaper"]
x=advertising[features]
y=advertising.Sales

from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=4,n_repeats=2,random_state=True)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(lm,x,y,cv=rkf)
print(scores)
print("Average score %.2f" % scores.mean())
#Ridge Regression
from sklearn.linear_model import Ridge

model_ridge = Ridge(alpha=0.5)
features = ["TV","Radio","Newspaper"]
x=advertising[features]
y=advertising.Sales

from sklearn.model_selection import cross_val_score
scores=cross_val_score(model_ridge,x,y,cv=5)
print(scores)
print("Average score %.2f" % scores.mean())
#Lasso Regression
from sklearn.linear_model import Lasso

model_ridge = Lasso(alpha=0.1)
features = ["TV","Radio","Newspaper"]
x=advertising[features]
y=advertising.Sales

from sklearn.model_selection import cross_val_score
scores=cross_val_score(model_ridge,x,y,cv=5)
print(scores)
print("Average score %.2f" % scores.mean())
#Elastic Net
from sklearn.linear_model import ElasticNet

model_elasticnet = ElasticNet(alpha=0.1,l1_ratio=0.5)
features = ["TV","Radio","Newspaper"]
x=advertising[features]
y=advertising.Sales

from sklearn.model_selection import cross_val_score
scores=cross_val_score(model_elasticnet,x,y,cv=5)
print(scores)
print("Average score %.2f" % scores.mean())