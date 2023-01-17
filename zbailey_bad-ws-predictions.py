import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Goal: Predict WS (World Series) outcome (Win/Loss) given playoff or WS team and regular season performance.

# Data: Teams.csv - db with every major league team from 1871-2016, various stats, and WSWin = Y or N.

# Method: Logistic regression on data with various numerical predictors

# Results: Odds are 50.9% that we correctly predict a WS teams outcome. Random guess is 50%.

teams = pd.read_csv("../input/Teams.csv")

teams.columns
# Filtering out seasons before the WS was established
teams = teams[teams['yearID']>=1903]

# Filtering out teams that did not make it to the world series.
teams = teams[teams['LgWin'] == "Y"]
# Assigning predictive variables for WS win
WS_Predictors = ['yearID','R','H','2B','3B','HR','BB','SO','SB','CG','ER','SHO','E','DP']
# Outcome vector: World series win Y/N
WS_Win = ['WSWin']
# Putting predictive and target variables in db for preprocessing
data = teams[WS_Predictors+WS_Win]

# Dropping NaN values
data = pd.get_dummies(data.dropna(axis=0))

# Replace categorical variables with dummy 0/1 variables/vectors
# Should probably change this
data = pd.get_dummies(data)

# Remove redundant column with 1/0 for WSWin = N/Y in favor of the opposite column.
data.drop('WSWin_N',inplace=True,axis=1)

# Separate predictors from target
X = data.loc[:, data.columns != 'WSWin_Y']
y = data['WSWin_Y']

## TODO ##
## 1. Try to make y data be the next years WS Win
## 2. Separate years in train/test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state = 15)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 
from sklearn.linear_model import LogisticRegression
# We assign an L1 penalty, this gives us a slightly better accuracy than l2 which is the default.
model = LogisticRegression(penalty = 'l1')
model.fit(X_train,y_train)
y_pred = model.decision_function(X_test)
y_int = model.predict(X_test)


test_inds = y_test.index

wins = data['WSWin_Y'].loc[test_inds]
teamyear = teams[['yearID','teamID']].loc[test_inds]
p = {'WS_Preds':y_pred}
preds = pd.DataFrame(data=p,index=y_test.index)
yp = {'WS_Preds':y_int}
preds_int = pd.DataFrame(data=yp,index=y_test.index)    
ytwp = pd.concat([teamyear,y_test,preds],axis=1)
ytwp = ytwp.sort_values(by=['yearID'])


for i in range(len(test_inds)-1):
    curyear = teamyear['yearID'].loc[test_inds[i]] 
    nextyear = teamyear['yearID'].loc[test_inds[i+1]]
    if curyear != nextyear:
        preds.loc[test_inds[i]] = np.heaviside(y_pred[i],0.5)
        preds.loc[test_inds[i+1]] = np.heaviside(y_pred[i+1],0.5)
    else:
        wini = preds.loc[test_inds[i]] > preds.loc[test_inds[i+1]]
        preds.loc[test_inds[i]] = wini
        preds.loc[test_inds[i+1]] = 1 - wini
        i = i+1
    
    
    
ytwp = pd.concat([teamyear,y_test,preds.astype(int),preds_int],axis=1)
ytwp = ytwp.sort_values(by=['yearID'])
print(ytwp)
# Odds that we correctly guess the WS winner and loser.

preds = np.array(preds).ravel()

odds = 1-np.sum(np.abs(preds-y_test))/preds.shape[0]

print(odds)
print(model.score(X_test,y_test))
# Notes: I need to change the model so when a winner or loser for given YEAR is guessed, 
# the other contender is automatically given the opposite label


