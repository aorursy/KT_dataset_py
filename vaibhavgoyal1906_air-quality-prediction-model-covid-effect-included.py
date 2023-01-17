# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/hyderabad-weather-with-air-quality-index-and-covid/Hyderabad-AirQ.csv')
df.head(20)
u= df['PM2.5'].max()
dp = df[['Lockdown','PM2.5']]
dp = dp.groupby('Lockdown').mean()
dp = dp.rename(columns={'PM2.5': 'PM2.5(Avg)'} , index={1 : 'Yes' , 0 : 'No'})
dp = dp.reset_index()
dp.head()
a = [0,1]
plt.rcParams['figure.figsize'] = (6,4)
k = plt.bar(dp['Lockdown'] , dp['PM2.5(Avg)'] , color='Red')
k[1].set_color('g')

plt.xlabel('Lockdown')
plt.ylabel('Avg PM2.5')
plt.legend()
target_column = ['PM2.5'] 
predictors = list(set(list(df.columns))-set(target_column))
predictors = list(set(list(df.columns))-set(['Date']))
df[predictors] = df[predictors]/df[predictors].max()
df.describe()
X = df[predictors].values
y = df[target_column].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(500, input_dim=9, activation= "relu"))
model.add(Dense(100, activation= "relu"))
model.add(Dense(50, activation= "relu"))
model.add(Dense(1))
model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
model.fit(X_train, y_train, epochs=20)
results = model.predict(X_test)
c = 0
for i in range(0 , len(results)):
  c = c+ (abs(results[i] - y_test[i])/y_test[i])*100
print('Accuracy is of')
print(100-(c/len(results))) 
xaxis = np.arange(1,len(results)+1)
y_test = np.array(y_test)
results = np.array(results)
y_te = y_test*u
res = results*u
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20,8)
plt.plot(xaxis, y_te , color='Red', label="Actual")
plt.plot(xaxis, res , color='Green', label="Predicted")
plt.legend()
plt.show()
s = pd.DataFrame(y_te)
s['results'] = res
s = s.rename(columns={0 : 'Actual' , 'results' :'Predicted'})
s['sn'] = xaxis
plt.bar(s['sn']+0.125, s['Actual'] ,width = 0.25, color='Red', label="Actual")
plt.bar(s['sn']-0.125, s['Predicted'] ,width = 0.25, color='Green', label="Predicted")
plt.legend()
plt.show()

s.head(46)