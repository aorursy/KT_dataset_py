# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/kaggle/input/car-data/CarPrice_Assignment.csv')
df['CarName'] = df['CarName'].apply(lambda x : x.split(" ")[0])
fueltype = pd.get_dummies(df['fueltype'],drop_first=True)
aspiration = pd.get_dummies(df['aspiration'],drop_first=True)
doornumber = pd.get_dummies(df['doornumber'],drop_first=True)
carbody = pd.get_dummies(df['carbody'],drop_first=True)
drivewheel = pd.get_dummies(df['drivewheel'],drop_first=True)
enginelocation = pd.get_dummies(df['enginelocation'],drop_first=True)
enginetype = pd.get_dummies(df['enginetype'],drop_first=True)
cylindernumber = pd.get_dummies(df['cylindernumber'],drop_first=True)
fuelsystem = pd.get_dummies(df['fuelsystem'],drop_first=True)
carnames = pd.get_dummies(df['CarName'],drop_first=True)
df.drop('fueltype',axis=1,inplace=True)
df.drop('aspiration',axis=1,inplace=True)
df.drop('doornumber',axis=1,inplace=True)
df.drop('carbody',axis=1,inplace=True)
df.drop('drivewheel',axis=1,inplace=True)
df.drop('enginelocation',axis=1,inplace=True)
df.drop('enginetype',axis=1,inplace=True)
df.drop('cylindernumber',axis=1,inplace=True)
df.drop('fuelsystem',axis=1,inplace=True)
df.drop('car_ID',axis=1,inplace=True)
df.drop('CarName',axis=1,inplace=True)
df = pd.concat([df,fueltype,aspiration,doornumber,carbody,drivewheel,enginelocation,enginetype,cylindernumber,fuelsystem,carnames],axis=1)
x = df[['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight',
       'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'gas',
       'turbo', 'two', 'hardtop', 'hatchback', 'sedan', 'wagon', 'fwd', 'rwd',
       'rear', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'rotor', 'five', 'four',
       'six', 'three', 'twelve', 'two', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi',
       'spdi', 'spfi', 'audi', 'bmw', 'buick', 'chevrolet', 'dodge', 'honda',
       'isuzu', 'jaguar', 'mazda', 'mercury', 'mitsubishi', 'nissan',
       'peugeot', 'plymouth', 'porsche', 'renault', 'saab', 'subaru', 'toyota',
       'volkswagen', 'volvo']]
y= df[['price']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=101)
from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor(n_estimators=10)
rfc.fit(x_train,y_train.values.ravel())
y_pred = rfc.predict(x_test)
print(r2_score(y_pred,y_test))
c = [i for i in range(len(y_pred))]
fig = plt.figure(figsize=(14,5))
plt.plot(c,y_test['price']-y_pred)
plt.xlabel('c', fontsize=18)                      # X-label
plt.ylabel('error', fontsize=16)                # Y-label
plt.show()
fig = plt.figure()
sns.distplot((y_test['price']-y_pred),bins=50)
plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label
plt.ylabel('Index', fontsize=16)                          # Y-label
plt.show()