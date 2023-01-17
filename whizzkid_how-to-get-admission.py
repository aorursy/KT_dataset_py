import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

df.drop(["Serial No."],inplace=True,axis=1)

df.columns=["gre","toefl","university_rating","sop","lor","cgpa","research","chance_of_admit"]

df.head(2)
df.info()
df_high=df[df.chance_of_admit>=0.90] # data with high probability of admission
plt.figure(figsize=(10,10))

plt.subplots_adjust(hspace=0.4,wspace=0.4)

for i in range(7):

    plt.subplot(4,2,i+1)

    sns.scatterplot(df_high.iloc[:,i],df_high.iloc[:,-1])
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor

from sklearn.linear_model import LinearRegression
classifiers={

    'support vector machine ':SVR(gamma='auto'),

    'decision tree          ':DecisionTreeRegressor(),

    'ada boost              ':AdaBoostRegressor(),

    'random forest          ':RandomForestRegressor(n_estimators=10),

    'linear regression      ':LinearRegression()

}
xdata=df.iloc[:,:-1].values

ydata=df.iloc[:,-1].values



xtrain,xtest,ytrain,ytest=train_test_split(xdata,ydata,test_size=0.20)

xtrain.shape,xtest.shape,ytrain.shape,ytest.shape
print("Model\t\t\t\t\t\tAccuracy\n")

for name,model in classifiers.items():

    model=model

    model.fit(xtrain,ytrain)

    score=model.score(xtest,ytest)

    print("{} :\t\t {}".format(name,score))