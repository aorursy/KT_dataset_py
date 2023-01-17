# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import dataset
car_df = pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv', encoding='ISO-8859-1')
car_df.head()
car_df.info()
# Visualize dataset
sns.pairplot(car_df)
# Correlation matrix (heatmap style)
corrmat = car_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#histogram and normal probability plot
sns.distplot(car_df['selling_price'], fit=norm);
fig = plt.figure()
res = stats.probplot(car_df['selling_price'], plot=plt)
#skewness and kurtosis
print("Skewness: %f" % car_df['selling_price'].skew())
print("Kurtosis: %f" % car_df['selling_price'].kurt())
#applying log transformation
car_df['selling_price'] = np.log(car_df['selling_price'])
#transformed histogram and normal probability plot
sns.distplot(car_df['selling_price'], fit=norm);
fig = plt.figure()
res = stats.probplot(car_df['selling_price'], plot=plt)
#scatter plot year/selling_price
var = 'year'
data = pd.concat([car_df['selling_price'], car_df[var]], axis=1)
data.plot.scatter(x=var, y='selling_price', ylim=(0,800000));
#scatter plot fuel/selling_price
var = 'fuel'
data = pd.concat([car_df['selling_price'], car_df[var]], axis=1)
data.plot.scatter(x=var, y='selling_price', ylim=(0,800000));
X = car_df.drop(['name', 'transmission', 'seller_type', 'fuel', 'owner','selling_price'], axis = 1)
X
y = car_df['selling_price']
y.shape
from sklearn.preprocessing import MinMaxScaler

scaler_x = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
scaler_x.data_max_
scaler_x.data_min_
print(X_scaled)
X_scaled.shape
X
y.shape
y = y.values.reshape(-1,1)
y.shape
y
scaler_y = MinMaxScaler()

y_scaled = scaler_y.fit_transform(y)
y_scaled
# Training the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)
# Let's import required libraries
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

model = Sequential()
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')
epochs_hist = model.fit(X_train, y_train, epochs=20, batch_size=25,  verbose=1, validation_split=0.2)
# Evaluating the model
print(epochs_hist.history.keys())
car_df.columns
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
# name, year, selling_price, km_driven, fuel, seller_type, transmission, owner  

# ***(Note that input data must be normalized)***

X_test_sample = np.array([[0, 0.4370344,  0.53515116, 0.57836085, 0.22342985]])
#X_test_sample = np.array([[1, 0.53462305, 0.51713347, 0.46690159, 0.45198622]])

y_predict_sample = model.predict(X_test_sample)

print('Expected selling price=', y_predict_sample)
y_predict_sample_orig = scaler_y.inverse_transform(y_predict_sample)
print('Expected selling price=', y_predict_sample_orig)
