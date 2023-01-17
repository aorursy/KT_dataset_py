# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline
import seaborn as sns
df = pd.read_csv('../input/housing.csv')
df.head()
df.isnull().sum()
df.isnull().sum()
sns.heatmap(df.isnull(),cmap='viridis',cbar=False,yticklabels=False)

plt.title('missing data')
plt.show()
df['total_bedrooms'].describe()
plt.figure(figsize=(10,4))
plt.hist(df[df['total_bedrooms'].notnull()]['total_bedrooms'],bins=20,color='green')#histogram of totalbedrooms
#data has some outliers
(df['total_bedrooms']>4000).sum()
plt.title('frequency historgram')
plt.xlabel('total bedrooms')
plt.ylabel('frequency')
# boxplot on total_bedrooms
plt.figure(figsize=(10,5))
sns.boxplot(y='total_bedrooms',data=df)
plt.plot
#we can see that area where median price frequencey for >= 500000 is more and could be a outlier or wrong data

plt.figure(figsize=(10,6))
sns.distplot(df['median_house_value'],color='purple')
plt.show()
df.info()
y_val = df['median_house_value']
y_val.isnull().sum()
y_val.head(10)
x_data = df.iloc[:,2:8]
x_data
x_data.head().T
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_data,y_val, test_size = 0.33, random_state = 101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = pd.DataFrame(data = scaler.transform(x_train),
                       columns = x_train.columns,index = x_train.index)
x_test = pd.DataFrame(data = scaler.transform(x_test),
                       columns = x_test.columns,index = x_test.index)
scaler.transform(x_train)
df.columns
age = tf.feature_column.numeric_column('housing_median_age')
rooms = tf.feature_column.numeric_column('total_rooms')
bedrooms = tf.feature_column.numeric_column('total_bedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('median_income')
feat_cols = [age,rooms,bedrooms,pop,households,income]
input_func = tf.estimator.inputs.pandas_input_fn(x = x_train,y = y_train,
                                batch_size = 200, num_epochs = 1000,shuffle = True)
model = tf.estimator.DNNRegressor(hidden_units=[6,6,6],feature_columns=feat_cols )
model.train(input_fn = input_func,steps = 20000)
