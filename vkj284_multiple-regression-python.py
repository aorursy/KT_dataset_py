import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
df = pd.read_csv('../input/addition/addition.csv')
df
x = df[['value1','value2','value3']]
y = df.output
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3)
abc = linear_model.LinearRegression()
abc.fit(xtrain,ytrain)
ypred = abc.predict(xtest)
NDS = pd.DataFrame()
NDS['value1'] = xtest['value1']
NDS['value2'] = xtest['value2']
NDS['value3'] = xtest['value3']
NDS['orgin - output'] = ytest.copy()
NDS['predict - output'] = ypred.copy()
NDS
abc.coef_
abc.intercept_
print(metrics.mean_absolute_error(ytest,ypred))
print(metrics.mean_squared_error(ytest,ypred))
print(np.sqrt(metrics.mean_squared_error(ytest,ypred)))
print((metrics.r2_score(ytest,ypred))*100)
abc.predict([[100,45,67]])