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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
df = pd.read_csv('/kaggle/input/wind-power-forecasting/Turbine_Data.csv')
df.info()
df["Unnamed: 0"] = df["Unnamed: 0"].apply(lambda x : datetime.strptime(x[:19],'%Y-%m-%d %H:%M:%S'))
df.describe()
drop_list = []
for i in df.columns:
    print(i,len(df[i].value_counts()))
    if len(df[i].value_counts()) == 1:
        drop_list.append(i)
df = df.drop(drop_list,axis = 1)
df = df.dropna()
plt.figure(figsize=(18,16))
sns.heatmap(df.corr(),square=True,annot=True,linewidths=0.1,cmap="coolwarm")
plt.show()
var = df.columns.values

i = 0

sns.set_style('whitegrid')
fig, ax = plt.subplots(5,4,figsize=(24,30))

for feature in var:
    if feature in ['WindSpeed','Unnamed: 0']:
        pass
    else:
        i += 1
        plt.subplot(5,4,i)
        sns.scatterplot(x=feature,y='WindSpeed', data=df[[feature,'WindSpeed']])
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
        
plt.show()
plt.figure(figsize=(50,5))
sns.lineplot(x='Unnamed: 0',y='WindSpeed', data=df[['Unnamed: 0','WindSpeed']])
plt.show()
from sklearn.preprocessing import RobustScaler, StandardScaler

rs = RobustScaler()
columns = df.columns.values.tolist()
columns.remove('Unnamed: 0')

preprocessed = rs.fit_transform(df[columns])
preprocessed = pd.DataFrame(preprocessed,columns=columns)

preprocessed['Time'] = pd.to_datetime(df['Unnamed: 0'].astype(str).values.tolist())

preprocessed.dtypes
def accuracy(predicted, observed):
    mse = abs(predicted - observed).mean()      # MSE, Mean Square Error
    rmse = ((predicted - observed)**2).mean()**.5  # RMSE, Root Mean Square Error
    mae = abs(predicted - observed).mean()      # MAE, Mean Absolute Error
    mape = abs((predicted - observed)/observed).mean()  # MAPE, Mean Absolute Percentage Error
    smape = (abs(predicted - observed)/((abs(predicted)+abs(observed))/2)).mean() # SMAPE, Symmetric Mean Absolute Percentage Error

    return({'MSE, Mean Square Error': mse, 
            'RMSE, Root Mean Square Error':rmse, 
            'MAE, Mean Absolute Error': mae, 
            'MAPE, Mean Absolute Percentage Error': mape , 
            'SMAPE, Symmetric Mean Absolute Percentage Error':smape})
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


column_list = ['ActivePower', 'AmbientTemperatue', 'BearingShaftTemperature',
       'Blade1PitchAngle', 'Blade2PitchAngle', 'Blade3PitchAngle',
       'GearboxBearingTemperature', 'GearboxOilTemperature', 'GeneratorRPM',
       'GeneratorWinding1Temperature', 'GeneratorWinding2Temperature',
       'HubTemperature', 'MainBoxTemperature', 'NacellePosition',
       'ReactivePower', 'RotorRPM', 'WindDirection']

x = preprocessed[column_list]

y = preprocessed['WindSpeed']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=100)
model = DecisionTreeRegressor().fit(train_x,train_y)
predict_y = model.predict(test_x)


for i in range(len(column_list)):
    print('%s: %.5f'%(column_list[i],model.feature_importances_[i]))
x = range(len(predict_y))
y1 = test_y.values
y2 = predict_y


plt.figure(figsize=(40,5))
sns.scatterplot(x=x, y=y1, legend= 'full')
sns.scatterplot(x=x, y=y2, legend= 'full')
plt.legend(['Accurency','Predicted'])
plt.show()
accuracy(y2, y1)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = preprocessed[column_list]

y = preprocessed['WindSpeed']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=100)

model = LinearRegression().fit(train_x, train_y)
predict_y = model.predict(test_x)

for i in range(len(column_list)):
    print('%s: %.5f'%(column_list[i], model.coef_[i]))
x = range(len(predict_y))
y1 = test_y.values
y2 = predict_y


plt.figure(figsize=(40,5))
sns.scatterplot(x=x, y=y1, legend= 'full')
sns.scatterplot(x=x, y=y2, legend= 'full')
plt.legend(['Accurency','Predicted'])
plt.show()
accuracy(y2, y1)