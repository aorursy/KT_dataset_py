# Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
admission = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
admission.info()
admission.head()
# Serial No. should no be taken into s
admission.drop(['Serial No.'],axis=1,inplace=True)
# Correlation between the variables

plt.figure(figsize=(8,6))
sns.heatmap(admission.corr(),annot=True,cmap='YlGnBu')
selected = admission[admission['Chance of Admit ']>0.8]
corrmat = selected.corr()
corrmat
corrmat["Chance of Admit "].sort_values(ascending=False)[1:]

# Finding the average parameters needed for admission in different rated universities

selected.groupby('University Rating').mean()
#University Rating vs GRE score
ax = selected.groupby('University Rating')['GRE Score'].mean().plot.bar()
ax.set_ylabel("GRE Score")
ax.set_ylim(320, 340)
# Training Data and Validation Data Split
traindata = admission.iloc[:400,]
valdata = admission.iloc[400:,]
valX = valdata.drop(["Chance of Admit "],axis=1)
valy = valdata["Chance of Admit "]
X = traindata.drop(["Chance of Admit "],axis=1)
y = traindata["Chance of Admit "]
# Train Test Split
from sklearn.model_selection import train_test_split

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.25,random_state=21)
# Lasso Linear Regression with hyperparamter tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso
lasso = Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lr_model = RandomizedSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5,random_state=21)
lr_model.fit(Xtrain,ytrain)
# Performance on test data
ypred1 = lr_model.predict(Xtest)
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt

rmse = round(sqrt(mean_squared_error(ytest,ypred1)),3)
r2score = round(r2_score(ytest,ypred1),3)

print(f'''RMSE Score of the model: {rmse}''')
print(f'''R2 Score of the model: {r2score}''')
# Performance on Validation Set
ypred2 = lr_model.predict(valX)
rmse = round(sqrt(mean_squared_error(valy,ypred2)),3)
r2score = round(r2_score(valy,ypred2),3)

print(f'''RMSE Score of the model: {rmse}''')
print(f'''R2 Score of the model: {r2score}''')
