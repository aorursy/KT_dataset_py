from sklearn.metrics import mean_squared_error

from sklearn import preprocessing

import itertools

from pprint import pprint

from sklearn.model_selection import RandomizedSearchCV

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNet

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor
df=pd.read_csv('train.csv')

df_test=pd.read_csv('test.csv')
df.drop(columns = ['id'], inplace = True)
test_df_agent = []



test_df_agent.append(df_test[df_test['a0'] == 1])

test_df_agent.append(df_test[df_test['a1'] == 1])

test_df_agent.append(df_test[df_test['a2'] == 1])

test_df_agent.append(df_test[df_test['a3'] == 1])

test_df_agent.append(df_test[df_test['a4'] == 1])

test_df_agent.append(df_test[df_test['a5'] == 1])

test_df_agent.append(df_test[df_test['a6'] == 1])



for i in range(0,7):

    test_df_agent[i].set_index('time',inplace = True)

    test_df_agent[i] = test_df_agent[i].drop(columns = ['a0','a1','a2','a3','a4','a5','a6'])  
df_agent = []



df_agent.append(df[df['a0'] == 1])

df_agent.append(df[df['a1'] == 1])

df_agent.append(df[df['a2'] == 1])

df_agent.append(df[df['a3'] == 1])

df_agent.append(df[df['a4'] == 1])

df_agent.append(df[df['a5'] == 1])

df_agent.append(df[df['a6'] == 1])



for i in range(0,7):

    df_agent[i].set_index('time',inplace = True)

    df_agent[i] = df_agent[i].drop(columns = ['a0','a1','a2','a3','a4','a5','a6'])   
for i in range(0,7):

    df_agent[i].drop(columns = ['b5','b10','b12','b16','b20','b25','b26','b28','b29','b33','b34','b35','b41','b42','b57','b58','b59','b61','b62','b72','b75','b77','b81','b89'],inplace = True)
X = []

Y = []

for i in range (0,7):

    Y.append(df_agent[i]['label'])

    X.append(df_agent[i].drop(columns = ['label']))
reg_rf = RandomForestRegressor()
for i in range(0,7):

    test_df_agent[i].drop(columns = ['b5','b10','b12','b16','b20','b25','b26','b28','b29','b33','b34','b35','b41','b42','b57','b58','b59','b61','b62','b72','b75','b77','b81','b89'],inplace = True)
output=[]

for i in range(0,7):

    reg_rf.fit(X[i],Y[i])

    prediction = reg_rf.predict(test_df_agent[i].drop(columns = ['id']))  

    output.append(pd.DataFrame({'id' : test_df_agent[i]['id'],'label' : prediction})) 
result = pd.concat(output)

result = result.sort_values(by=['id'])
result
submissionfile='varadsubmission1.csv'

result.to_csv(submissionfile,index=False)
gradboost = GradientBoostingRegressor()

output2=[]

for i in range(0,7):

    gradboost.fit(X[i],Y[i])

    prediction = gradboost.predict(test_df_agent[i].drop(columns = ['id']))  

    output2.append(pd.DataFrame({'id' : test_df_agent[i]['id'],'label' : prediction})) 
result2 = pd.concat(output)

result2= result2.sort_values(by=['id'])
result2
submissionfile2='varadsubmission2.csv'

result2.to_csv(submissionfile2,index=False)