# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.getcwd())
#os.chdir('/Users/xianglongtan/Desktop/kaggle')
print(os.listdir("../input"))
#print(os.listdir())
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_activity = 'all'
# Any results you write to the current directory are saved as output.
app_train = pd.read_csv('../input/application_train.csv')
#app_train = pd.read_csv('application_train.csv')
#app_train.head()
app_test = pd.read_csv('../input/application_test.csv')
#app_test = pd.read_csv('application_test.csv')
#app_test.head()
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xgboost import XGBClassifier
def Imputation(df, col, n=10):
    Y = df[col]
    X = pd.get_dummies(df.drop(col,axis=1))
    datatype = Y.dtype
    if datatype == 'float64':
        model = XGBRegressor()
    elif datatype == 'object' or 'int64':
        model = XGBClassifier()
    obs = pd.isnull(df[col]) != True
    mis = pd.isnull(df[col]) == True
    model.fit(X[obs],Y[obs])
    df.loc[mis,col] = model.predict(X.loc[mis])
    for i in range(n):
        Y = df[col]
        #print(X)
        #print(Y)
        #print(model)
        model.fit(X,Y)
        df[mis][col] = model.predict(X[mis])
    return df
import numpy as np
a = pd.DataFrame([[1,'a','cd'],[2,'b','ef'],[3,np.nan,'gh'],[np.nan,'c','ij'],[5,'d',np.nan],[np.nan,'e',np.nan],[6,np.nan,np.nan],[np.nan,np.nan,np.nan]], columns = ['X1','X2','Y'])
b = pd.DataFrame([[2,'a'],[3,'c']],columns=['X1','X2'])
Imputation(a,'X2')
