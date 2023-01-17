%matplotlib inline
import pandas as pd

import numpy as np  

import matplotlib.pyplot as plt 

import matplotlib.dates 

import datetime 

%matplotlib inline

train = pd.read_csv('../input/train.csv')

store = pd.read_csv('../input/store.csv')

test = pd.read_csv('../input/test.csv') 



store.head()
train.head()
test.head()
## transform date variable

datetimes = [datetime.datetime.strptime(t, "%Y-%m-%d") for t in train.Date]

plotData = matplotlib.dates.date2num(datetimes) 

train = train.join(pd.DataFrame(plotData,columns = ['datetimes']))

def splitTime(x): 

    mysplit = datetime.datetime.strptime(x,  "%Y-%m-%d") 

    return [mysplit.year,mysplit.month,mysplit.day]

train = train.join(pd.DataFrame(train.Date.apply(splitTime).tolist(), columns = ['year','mon','day']))



# plot the first 5 stores sales vs time



for i in range(1,5):

 plt.figure(i,figsize=(20,10)) 

 plt.subplot(211)

 plt.plot_date(train.loc[train.Store==i,'datetimes'],train.loc[train.Store==i,'Sales'],linestyle='-') 

 plt.title('Store %d' %i)  

 plt.subplot(212)

 train2014 = train.loc[train.year == 2014,:]

 plt.plot_date(train2014.loc[train2014.Store==i,'datetimes'],train2014.loc[train2014.Store==i,'Sales'],linestyle='-') 

 plt.title('Store %d, 2014' %i)  

 plt.show()
## distribution of sales variable

plt.figure(1,figsize=(15,10)) 

plt.subplot(221)

plt.hist(train.Sales,bins=30)

plt.title("Distribution of Sales") 

plt.subplot(222)

plt.hist(np.log(train.Sales+1),bins=30)

plt.title("Distribution of log(Sales)") 
## average log sales, by store



plt.hist([np.log(train.groupby('Store').Sales.mean()) ],bins=30)
toAppend = pd.DataFrame(np.log(train.Sales+1),dtype=float)

toAppend.columns.values[0]='LogSale'

train=train.join(toAppend)

train.dtypes
### data transformation on store data set



## transform variable PromoInterval to 12 dummy variables

def myPinterval(x):

    if x=='Feb,May,Aug,Nov':  return([0,1,0,0,1,0,0,1,0,0,1,0])

    elif x=='Jan,Apr,Jul,Oct':  return([1,0,0,1,0,0,1,0,0,1,0,0])

    elif x== 'Mar,Jun,Sept,Dec': return([0,0,1,0,0,1,0,0,1,0,0,1])

    else: return(np.repeat(0,12).tolist())



proInt = store.PromoInterval.apply(myPinterval).tolist()

proInt = pd.DataFrame(proInt, columns = ['ProInt'+ str(i) for i in range(1,13)])

store = store.drop('PromoInterval',1).join(proInt)



store = store.drop('StoreType',1).join(pd.get_dummies(store['StoreType']).rename(columns=lambda x: 'StoreType' +"_"+str(x)))  

store = store.drop('Assortment',1).join(pd.get_dummies(store['Assortment']).rename(columns=lambda x: 'Assortment' +"_"+str(x)))  

##assume 0 and '0' are the same in train.StateHoliday 

def mychange(x):

     if type(x)!= str: x=str(x)

     return x

        

train.StateHoliday = [mychange(x) for x in train.StateHoliday]



newtrain = train.drop('StateHoliday',1).join(pd.get_dummies(train['StateHoliday']).rename(columns=lambda x: 'StateHoliday' +"_"+str(x)))  
## merge training set with store



newtrain=pd.merge(newtrain, store, on="Store")  

newtrain.drop(['Date','Customers','datetimes','Sales'],axis = 1,inplace=True)
## do the same thing on testing set

test = test.join(pd.DataFrame(test.Date.apply(splitTime).tolist(), columns = ['year','mon','day']))

newtest = test.drop('StateHoliday',1).join(pd.get_dummies(test['StateHoliday']).rename(columns=lambda x: 'StateHoliday' +"_"+str(x)))  

newtest = pd.merge(newtest,store, on="Store")

newtest.drop(['Date'],axis = 1,inplace=True) 
## check if there exists any constant variable

np.sum(newtrain.var()==0)
##### randomforest

from sklearn.ensemble.forest import RandomForestRegressor



##### delete variables that do not exist in the test set

toDrop = list(set(newtrain.columns.values)-set(newtest.columns.values) )

features = newtrain.columns.drop(toDrop,1)



rf = RandomForestRegressor(n_estimators=100)

rf.fit(newtrain.drop(toDrop ,1).fillna(-1),newtrain.LogSale)





importances = rf.feature_importances_ 

# return the indices that would sort the importance, decreasing

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")





Features = newtrain.columns.drop('LogSale')

for f in range(35):

    print("%d. feature %d :%s (%f)" % (f + 1, indices[f],Features[indices[f]], importances[indices[f]]))



# Plot the feature importances of the forest

# the most important feature 'open' is left out in the plot to make it easier to see the other features

plt.figure()

plt.title("Feature importances")

plt.bar(range(1,10), importances[indices[range(1,10)]]) 

plt.xlim([-1, 10])

plt.show()



# make prediction on test data

mypred = rf.predict(newtest.drop('Id',1).fillna(-1))
mypred = np.exp(mypred)-1

mypred = pd.DataFrame({ 'Id': test['Id'],

                            'Sales': mypred[np.argsort(newtest['Id'])] })

#mypred.to_csv("randomForest_1stSubmission.csv", index=False)
############lasso for prediction 

import pasty

from sklearn import linear_model

y,X  = patsy.dmatrices("LogSale ~ 1+sx+rk+yr+dg+yd",newtrain)



newtrain.columns







alphas = np.logspace(-4, -.5, 30)