import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import math
%matplotlib inline

np.random.seed(0)

df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')

#df_train = df_train[(df_train['rating']!=6)]
#df_train = df_train[(df_train['rating']!=0)]

#df_train.fillna(value=df_train.mean(),inplace=True)
df_train.dropna(inplace = True)
y_train = df_train['rating']
df_train = pd.get_dummies(data = df_train, columns=['type'])
df_train = df_train.drop(columns = ['rating','type_old'])

df_test_2 = df_test.copy()
df_test_2 = pd.get_dummies(data = df_test_2, columns=['type'])
id_test = df_test['id']
df_test.fillna(value=df_test.mean(),inplace=True)
df_test = pd.get_dummies(data = df_test, columns=['type'])
df_test = df_test.drop(columns = ['type_old'])


from sklearn.model_selection import train_test_split

x_train_val,x_test_val,y_train_val,y_test_val = train_test_split(df_train,y_train,test_size=0.3,random_state=27)

i = 200
j = 20
k = 27
from sklearn.ensemble import ExtraTreesRegressor
regressor = ExtraTreesRegressor(n_estimators=i,n_jobs=-1,random_state=k,max_depth=j)
regressor.fit(x_train_val, y_train_val)

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test_val,[round(s) for s in regressor.predict(x_test_val)]))
from sklearn.ensemble import ExtraTreesRegressor
regressor = ExtraTreesRegressor(n_estimators=i,n_jobs=-1,random_state=27,max_depth=j)
regressor.fit(df_train, y_train)
print(type(df_test))
df_test['rating'] = regressor.predict(df_test).reshape((-1,1))
df_f = df_test_2.join(df_test['rating'],how='left')
df_f.fillna(value=df_f.mean(),inplace=True)
df_f['rating'] = np.rint(df_f['rating'])
out = pd.concat([id_test,df_f['rating']],axis=1)
out.columns = ['id','rating']
out.to_csv('final1.csv', index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import math
%matplotlib inline

np.random.seed(0)

df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')

#df_train = df_train[(df_train['rating']!=6)]
#df_train = df_train[(df_train['rating']!=0)]

#df_train.fillna(value=df_train.mean(),inplace=True)
df_train.dropna(inplace = True)
y_train = df_train['rating']
df_train = pd.get_dummies(data = df_train, columns=['type'])
df_train = df_train.drop(columns = ['rating','type_old'])

df_test_2 = df_test.copy()
df_test_2 = pd.get_dummies(data = df_test_2, columns=['type'])
id_test = df_test['id']
df_test.fillna(value=df_test.mean(),inplace=True)
df_test = pd.get_dummies(data = df_test, columns=['type'])
df_test = df_test.drop(columns = ['type_old'])


from sklearn.model_selection import train_test_split

x_train_val,x_test_val,y_train_val,y_test_val = train_test_split(df_train,y_train,test_size=0.3,random_state=27)

i = 2000
k = 27
from sklearn.ensemble import ExtraTreesRegressor
regressor = ExtraTreesRegressor(n_estimators=i,n_jobs=-1,random_state=k)
regressor.fit(x_train_val, y_train_val)

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test_val,[round(s) for s in regressor.predict(x_test_val)]))
from sklearn.ensemble import ExtraTreesRegressor
regressor = ExtraTreesRegressor(n_estimators=i,n_jobs=-1,random_state=27)
regressor.fit(df_train, y_train)
print(type(df_test))
df_test['rating'] = regressor.predict(df_test).reshape((-1,1))
df_f = df_test_2.join(df_test['rating'],how='left')
df_f.fillna(value=df_f.mean(),inplace=True)
df_f['rating'] = np.rint(df_f['rating'])
out = pd.concat([id_test,df_f['rating']],axis=1)
out.columns = ['id','rating']
out.to_csv('final2.csv', index=False)


