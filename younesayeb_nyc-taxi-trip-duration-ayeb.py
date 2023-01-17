import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

print(os.listdir("../input"))

%matplotlib inline
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

test.info()

train.info()
train.describe()
train.dtypes

plt.subplots(figsize=(14,5))

plt.title("Outliers visualization")

train.boxplot();
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
train = train[train['passenger_count']> 0]

train
x = train[["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude"]]
y= train['trip_duration']

y.describe()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

x_train.shape, y_train.shape, x_test.shape, y_test.shape
rand = RandomForestRegressor()
rand.fit(x_train,y_train)
x_test = test[["vendor_id", "passenger_count","pickup_longitude", "pickup_latitude","dropoff_longitude","dropoff_latitude"]]

prediction = rand.predict(x_test)

prediction
sub = pd.read_csv("../input/sample_submission.csv")

sub.head()
submission = pd.DataFrame({'id': test.id, 'trip_duration': prediction})

submission.head()
submission.to_csv('submission.csv', index=False)