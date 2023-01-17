import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
test = pd.read_csv("../input/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
Xtrain = train.iloc[:,1:-1] 
Ytrain = train['median_house_value']
Xtest = test.iloc[:,1:-1]
train.iloc[0:20]
train.shape
train[train.columns].corr()["median_house_value"]
train.std()
train.std()/train.mean()
train['total_rooms'].plot(kind = 'hist', bins = 50, xlim=(-500,20000))
train['total_bedrooms'].plot(kind = 'hist', bins = 50, xlim=(-500,4000))
train['population'].plot(kind = 'hist', bins = 80, xlim=(-500,10000))
train['households'].plot(kind = 'hist', bins = 60, xlim=(-100,3500))
from sklearn.metrics import make_scorer
def RMSLE(Y,Ypred):
    soma = 0
    Y = np.array(Y)
    for i in range(0,len(Ypred)):
        soma += (np.log(1+abs(Ypred[i]))-np.log(Y[i]+1))**2
    rmsle = np.sqrt(soma/len(Y))
    return rmsle
scorer = make_scorer(RMSLE)
#greater_is_better = False
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
scores = {}
for k in range(1,80):
    knr = KNeighborsRegressor(n_neighbors = k)
    knr.fit(Xtrain,Ytrain)
    Ypred_train = knr.predict(Xtrain)
    score = RMSLE(Ytrain,Ypred_train) 
    scores[k]=score
scores
Ypred = knr.predict(Xtrain)
cv_scores = cross_val_score(knr, Xtrain, Ytrain, cv=10, scoring= scorer)
scores = {}
for k in range(1,80):
    knr = KNeighborsRegressor(n_neighbors = k)
    knr.fit(Xtrain,Ytrain)
    Ypred = knr.predict(Xtrain)
    cv_scores = cross_val_score(knr, Xtrain, Ytrain, cv=10, scoring= scorer)
    scores[k]=np.mean(cv_scores)
scores
min(scores,key=scores.get)
scores = {}
for p in range(1,10):
    knr = KNeighborsRegressor(n_neighbors = 12,p=p)
    knr.fit(Xtrain,Ytrain)
    Ypred = knr.predict(Xtrain)
    cv_scores = cross_val_score(knr, Xtrain, Ytrain, cv=10, scoring= scorer)
    scores[p]=np.mean(cv_scores)
scores
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
lr = LinearRegression()
lr.fit(Xtrain,Ytrain)
Y_pred = lr.predict(Xtrain)
cv_scores = cross_val_score(lr, Xtrain, Ytrain, cv=10, scoring= scorer)
cv_scores
cv_scores.mean()
lr = Lasso()
lr.fit(Xtrain,Ytrain)
Y_pred = lr.predict(Xtrain)
cv_scores = cross_val_score(lr, Xtrain, Ytrain, cv=10, scoring= scorer)
cv_scores
cv_scores.mean()
ridge = Ridge()
ridge.fit(Xtrain,Ytrain)
Y_pred = lr.predict(Xtrain)
cv_scores = cross_val_score(ridge, Xtrain, Ytrain, cv=10, scoring= scorer)
cv_scores
cv_scores.mean()
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(Xtrain,Ytrain)
Y_pred = rfr.predict(Xtrain)
cv_scores = cross_val_score(rfr, Xtrain, Ytrain, cv=10, scoring= scorer)
cv_scores
cv_scores.mean()
scores = {}
for i in range(1,35):
    rfr = RandomForestRegressor(n_estimators = i)
    rfr.fit(Xtrain,Ytrain)
    Y_pred = rfr.predict(Xtrain)
    cv_scores = cross_val_score(rfr, Xtrain, Ytrain, cv=10, scoring= scorer)
    scores[i] = cv_scores 
for i in scores:
    print(scores[i].mean(),' ........ ',i)
rfr = RandomForestRegressor(n_estimators = 70)
rfr.fit(Xtrain,Ytrain)
Y_pred = rfr.predict(Xtrain)
cv_scores = cross_val_score(rfr, Xtrain, Ytrain, cv=10, scoring= scorer)
cv_scores.mean()
rfr = RandomForestRegressor(n_estimators = 50)
rfr.fit(Xtrain,Ytrain)
Y_pred = rfr.predict(Xtrain)
cv_scores = cross_val_score(rfr, Xtrain, Ytrain, cv=10, scoring= scorer)
cv_scores.mean()
rfr = RandomForestRegressor(n_estimators = 100)
rfr.fit(Xtrain,Ytrain)
Y_pred = rfr.predict(Xtrain)
cv_scores = cross_val_score(rfr, Xtrain, Ytrain, cv=10, scoring= scorer)
cv_scores.mean()
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
AdaBoost = AdaBoostRegressor(base_estimator = dtr, n_estimators = 10, loss = 'linear', learning_rate = 1.0)
AdaBoost.fit(Xtrain,Ytrain)
Y_pred = AdaBoost.predict(Xtrain)
cv_scores = cross_val_score(AdaBoost, Xtrain, Ytrain, cv=10, scoring= scorer)
cv_scores.mean()
scores = {}
for n in range(1,200,10):
    print(n)
    dtr = DecisionTreeRegressor()
    AdaBoost = AdaBoostRegressor(base_estimator = dtr, n_estimators = n, loss = 'linear', learning_rate = 1.0)
    AdaBoost.fit(Xtrain,Ytrain)
    Y_pred = AdaBoost.predict(Xtrain)
    cv_scores = cross_val_score(AdaBoost, Xtrain, Ytrain, cv=10, scoring= scorer)
    scores[n] = cv_scores
scores_medios = {}
for i in scores:
    print(scores[i].mean(),' ........ ',i)
    scores_medios[i] = scores[i].mean()
lists = sorted(scores_medios.items())
plt.figure(figsize=(10,10))
x, y = zip(*lists)

plt.plot(x, y, 'ro')
plt.show()
dtr = DecisionTreeRegressor()
AdaBoost = AdaBoostRegressor(base_estimator = dtr, n_estimators = 200, loss = 'linear', learning_rate = 1.0)
AdaBoost.fit(Xtrain,Ytrain)
Y_pred = AdaBoost.predict(Xtrain)
cv_scores = cross_val_score(AdaBoost, Xtrain, Ytrain, cv=10, scoring= scorer)
cv_scores.mean()