#CMPE353-FinalProject
#Bihter ÇAKAL
#Veyis Egemen ERDEN
import numpy as np 
import pandas as pd 

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#dropped id 
train = train.drop('Id', axis=1)
test = test.drop('Id', axis=1)
#The section we used to calculate how many years ago the restaurant was opened.
#converted opendate format for to process
train['Open Date'] = pd.to_datetime(train['Open Date'], format='%m/%d/%Y')
#Train dataFrame
dateLastTrain = pd.DataFrame({'Date':np.repeat(['05/19/2018'],[len(train)]) })
dateLastTrain['Date'] = pd.to_datetime(dateLastTrain['Date'], format='%m/%d/%Y')
#converted day to year
train['Years'] = (dateLastTrain['Date'] - train['Open Date']) / 365
train['Years'] = train['Years'].astype('timedelta64[D]').astype(int)
#we do not need Open Date column anymore.           
train = train.drop('Open Date', axis=1)

#Test dataFrame
test['Open Date'] = pd.to_datetime(test['Open Date'], format='%m/%d/%Y')
test['Years']=""
#converted day to year
dateLastTest = pd.DataFrame({'Date':np.repeat(['05/19/2018'],[len(test)]) })
dateLastTest['Date'] = pd.to_datetime(dateLastTest['Date'], format='%m/%d/%Y')  
        
test['Years'] = (dateLastTest['Date'] - test['Open Date']) / 365
test['Years'] = test['Years'].astype('timedelta64[D]').astype(int)
#we do not need Open Date column anymore.                      
test = test.drop('Open Date', axis=1)


#converted categorical data to numaric data for process.
#Train dataframe
cityGrouptoNumaricTypeTrain = pd.get_dummies(train['City Group'])
train = train.join(cityGrouptoNumaricTypeTrain)
#Test dataframe
cityGrouptoNumaricTypeTest = pd.get_dummies(test['City Group'])
test = test.join(cityGrouptoNumaricTypeTest)

train = train.drop('City Group', axis=1)
test = test.drop('City Group', axis=1)
#converted categorical data to numaric data for process.
#Train dataframe
cityTypetoNumaricTypeTrain = pd.get_dummies(train['Type'])
train = train.join(cityTypetoNumaricTypeTrain)

train = train.drop('Type', axis=1)
#Test dataframe
cityTypetoNumaricTypeTest = pd.get_dummies(test['Type'])
test = test.join(cityTypetoNumaricTypeTest)

test = test.drop('Type', axis=1)



#Standardization Features we use for features P-1 to P-37 and the points of the properties are summed and written as a new column
from sklearn import preprocessing

#Train dataFrame
x = train.iloc[:, 1:38] #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x).sum(axis=1) 
train['sumOfFeatures']= x_scaled

#Test dataFrame
x = test.iloc[:, 1:38] #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler.fit_transform(x).sum(axis=1) 
test['sumOfFeatures']= y_scaled




import seaborn as sns
#the earnings of restaurants according to their age.
yearRevenue = train[["Years","revenue"]].groupby(['Years'],as_index=False).mean()
data = yearRevenue.sort_values(["revenue"],ascending= False)
sns.barplot(x='Years', y='revenue', data=data)
#the earnings of restaurants according to where the restaurants are.
cityRevenue = train[["City","revenue"]].groupby(['City'],as_index=False).mean()
data = cityRevenue.sort_values(["revenue"],ascending= False)
#the first six
sns.barplot(x='City', y='revenue', data=data.head(6))
#the last six
sns.barplot(x='City', y='revenue', data=data.tail(6)).invert_xaxis()
#Normalized
#from sklearn import preprocessing


#x = train.iloc[:, 1:38]

#X_normalized = preprocessing.Normalizer().fit(x) 
#train.loc[:,'ColumnA']= X_normalized.transform(x).sum(axis=1) 

#y = test.iloc[:, 1:38]
#y_normalized = preprocessing.Normalizer().fit(y) 
#test.loc[:,'ColumnA']= y_normalized.transform(y).sum(axis=1) 





from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

#created dataframe with Year,(FC,IL)Type,ColumnA(sum of features P-1 to P-37),Big Cities,Other
import numpy
xTrain = pd.DataFrame({'Years':train['Years'],'FC':train['FC'],'IL':train['IL'],'sumOfFeatures':train['sumOfFeatures'],
                      'Big Cities':train['Big Cities'], 'Other':train['Other']})

yTrain = train['revenue'].apply(numpy.log)
xTest = pd.DataFrame({'Years':test['Years'],'FC':test['FC'],'IL':test['IL'],'sumOfFeatures':test['sumOfFeatures'],
                      'Big Cities':test['Big Cities'], 'Other':test['Other']})


#create model estimators 230 it works better
cls = RandomForestRegressor(n_estimators=230)
cls.fit(xTrain, yTrain)

pred = cls.predict(xTest)

pred = numpy.exp(pred)

cls.score(xTrain, yTrain)
test['revenue']=pred.astype(int)
test.head(10)