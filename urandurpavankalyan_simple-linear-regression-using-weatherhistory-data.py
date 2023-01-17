import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

class LinearRegression:
    def __init__(self,X,Y):
        ones=np.ones(X.shape)
        X=np.append(ones,X,axis=1)
        self.X=X
        self.Y=Y
        self.m=X.shape[0]
        self.n=X.shape[1]
        self.theta=np.random.randn(X.shape[1])
        
    def computeCostFunction(self):
        h=np.matmul(self.X,self.theta)
        self.J=(1/(2*self.m))*np.sum((h-self.Y)**2)
        return self.J
    
    def performGradientDescent(self,num_of_iter,alpha):
        self.Cost_history=[]
        self.theta_history=[]
        for x in range(num_of_iter):
            h=np.matmul(self.X,self.theta)
            J=self.computeCostFunction()
            self.Cost_history.append(J)
            self.theta_history.append(self.theta)
            temp=h-self.Y
            self.theta=self.theta-(alpha/self.m)*(self.X.T.dot(temp))
        return self.theta,self.Cost_history,self.theta_history
            
        
    def predict(self,X_test,Y_test):
        ones=np.ones(X_test.shape)
        X_test=np.append(ones,X_test,axis=1)
        self.Y_pred=np.matmul(X_test,self.theta)
        self.error=self.Y_pred-Y_test
        return self.Y_pred,self.error
    
    def predictUsingNormalEquation(self,X_test,Y_test):
        ones=np.ones(X_test.shape)
        X_test=np.append(ones,X_test,axis=1)
        inv=np.linalg.inv(np.matmul(self.X.T,self.X))
        self.w=np.matmul(np.matmul(inv,self.X.T),self.Y)
        y_pred=np.matmul(X_test,self.w)
        return y_pred,(Y_test-y_pred)
        
        
    
        
    def returnTheta(self):
        return self.theta
    
    def returnX(self):
        return self.X
        
    def returnY(self):
        return self.Y
        
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


#This note book does two tasks
"""
1. Given humidity predict temperature
2. Given humidity predict apparent temperature
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt;

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#reading the data set
dataset=pd.read_csv('../input/szeged-weather/weatherHistory.csv') #read the data set weatherHistory.csv
humidity=dataset.iloc[:,5:6].values #load the humidity column to humidity np array
temperature=dataset.iloc[:,3:4].values #load the temperature column to temperature np array
apparentTemperature=dataset.iloc[:,4:5].values #load the apparent Temperature column to apparentTemperature np array
dataset.describe()



#replace missing data with mean using imputer library
from sklearn.preprocessing import Imputer
imputerHumidity=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputerHumidity=imputerHumidity.fit(humidity) 
humidity=imputerHumidity.transform(humidity) #if any missing data exists in the humidity np array, it will be replaced by mean humidity

imputerTemperature=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputerTemperature=imputerTemperature.fit(temperature)
temperature=imputerTemperature.transform(temperature) #if any missing data exists in the temperature np array, it will be replaced by mean temperature


imputerappTemp=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputerappTemp=imputerappTemp.fit(apparentTemperature)
apparentTemperature=imputerappTemp.transform(apparentTemperature) #if any missing data exists in the apparentTemperature np array, it will be replaced by mean apparentTemperature


#split the data into training and testing
from sklearn.cross_validation import train_test_split
humidity_train,humidity_test,temperature_train,temperature_test=train_test_split(humidity,temperature,test_size=0.2,random_state=0)
humidity_train,humidity_test,apptemp_train,apptemp_test=train_test_split(humidity,apparentTemperature,test_size=0.2,random_state=0)
temperature_train=temperature_train.flatten()
temperature_test=temperature_test.flatten()
apptemp_train=apptemp_train.flatten()
apptemp_test=apptemp_test.flatten()
lr1=LinearRegression(humidity_train,temperature_train)
print(lr1.computeCostFunction())
theta1=lr1.returnTheta()
theta1,cost_history1,theta_history1=lr1.performGradientDescent(10000,0.01)
temperature_predict,temperature_error=lr1.predict(humidity_test,temperature_test)
temperature_pred_normal,error_temp_normal=lr1.predictUsingNormalEquation(humidity_test,temperature_test)

plt.scatter(humidity_test,temperature_test)
plt.plot(humidity_test,temperature_pred_normal,'r')
plt.title('Humidity vs Temperature using normal equation')
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.show()


plt.scatter(humidity_test,temperature_test)
plt.plot(humidity_test,temperature_predict,'r')
plt.title('Humidity vs Temperature using gradient descent')
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.show()
lr2=LinearRegression(humidity_train,apptemp_train)
print(lr2.computeCostFunction())
theta2=lr2.returnTheta()
theta2,cost_history2,theta_history2=lr2.performGradientDescent(100000,0.01)
apptemp_predict,apptemp_error=lr2.predict(humidity_test,apptemp_test)
apptemp_pred_normal,error_apptemp_normal=lr2.predictUsingNormalEquation(humidity_test,apptemp_test)

plt.scatter(humidity_test,apptemp_test)
plt.plot(humidity_test,apptemp_pred_normal,'r')
plt.title('Humidity vs Apparent Temperature using normal equation')
plt.xlabel('Humidity')
plt.ylabel('Apparent Temperature')
plt.show()

plt.scatter(humidity_test,temperature_test)
plt.plot(humidity_test,temperature_predict,'r')
plt.title('Humidity vs Apparent Temperature using gradient descent')
plt.xlabel('Humidity')
plt.ylabel('Apparent Temperature')
plt.show()