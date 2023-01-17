# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import sys

from pandas import Series

import traceback

import time

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import explained_variance_score

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from pandas.tools.plotting import scatter_matrix

df = pd.DataFrame(columns=['year_quter','Air_Humidity(%)','Air_Pressure(hPa)','Air_Temperature(°C)','Dew_Point','Rain_Cum','Rain_Value(mm)','Rain_Cumulative(mm)','Soil_Moisture1','Soil_Moisture2','Soil_Temperature1','Soil_Temperature2',

'Solar_Radiation','Sunshine_Count','Wind_Chill','Wind_Direction(Degree)','Wind_Speed(kph)','Wind_Speed_max'])

iteri=0

degrees = 0

Air_Humidity = 0

Air_Pressure = 0

Air_Temperature = 0

Dew_Point = 0

Rain_Cum = 0

Rain_Value = 0

Rain_Cumulative = 0

Soil_Moisture1 = 0

Soil_Moisture2 = 0

Soil_Temperature1 = 0

Soil_Temperature2 = 0

Solar_Radiation = 0

Sunshine_Count = 0

Wind_Chill = 0

Wind_Direction = 0

Wind_Speed = 0

Wind_Speed_max = 0



global iteri

global df
data  = pd.read_csv('../input/2018-1_4 - DATA.csv') #data= weather Information

data.info()
data.head(10)
data.drop('price',axis=1,inplace=True)
data.head(10)
print('Shape of the data set: ' + str(data.shape))
print("describe: ")

print(data.describe())
fig = data.hist(bins=50, figsize=(20,15))
corr_matrix = data.corr()

attributes = ['Air_Humidity(%)','Air_Temperature(°C)','Soil_Moisture1','Soil_Temperature1','Solar_Radiation_()']

fig = scatter_matrix(data[attributes], figsize = (10,10), alpha = 1)
data.dropna(inplace=True)

data.drop_duplicates(inplace=True)

sns.pairplot(data);
plt.figure(figsize=(20,10)) 

sns.heatmap(data.corr(),annot=True, cmap='cubehelix_r') 

plt.show()
data.head(10)
data2 = data[['Air_Humidity(%)','Air_Pressure(hPa)','Air_Temperature(°C)','Dew_Point','Rain_Cum','Rain_Value(mm)','Rain_Cumulative(mm)','Soil_Moisture1','Soil_Moisture2','Soil_Temperature1','Soil_Temperature2',

'Solar_Radiation','Sunshine_Count','Wind_Chill','Wind_Direction(Degree)','Wind_Speed(kph)','Wind_Speed_max','price','Production(tons)']]

X = data2.drop('Production(tons)',axis=1).values

print(X)

y = data2['Production(tons)'].values

print(y)



data2.head()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)
neighbors = np.arange(1,9)

print(neighbors)



# 값을 저장할 배열을 미리 만들어 둠. 생성된 배열 값은 기존 메모리에 있는 값이 들어 있어서

# 이상한 값이 들어 있을 수 있음.

train_accuracy = np.empty(len(neighbors))

print(train_accuracy)



test_accuracy = np.empty(len(neighbors))

print(train_accuracy)



for i, k in enumerate(neighbors):

    print(i, k)

    knn = KNeighborsClassifier(n_neighbors=k)

    

    knn.fit(X_train, y_train)

    

    train_accuracy[i] = knn.score(X_train, y_train)    

    test_accuracy[i] = knn.score(X_test, y_test)

    print(train_accuracy[i], test_accuracy[i])
plt.figure(figsize=(10,7)) 

plt.title('k-NN Varying number of neighbors')

plt.plot(neighbors, test_accuracy, label='Testing Accuracy')

plt.plot(neighbors, train_accuracy, label='Training accuracy')

plt.legend()

plt.xlabel('Number of neighbors')

plt.ylabel('Accuracy')

plt.show()
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

score = knn.score(X_test,y_test)

print(score)



# 실제로 예측한 값

y_pred = knn.predict(X_test)

print('예측한 값:', y_pred)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)