import tensorflow as tf

from tensorflow import keras



import os

import tempfile



import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



import sklearn

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler,RobustScaler



from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.layers import Conv2D, MaxPooling2D, Flatten

from keras.optimizers import SGD, Adam

from keras.utils import np_utils
file = tf.keras.utils

raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')

raw_df.head()
raw_df.describe()
raw_df.isnull().sum().max()
neg, pos = np.bincount(raw_df['Class'])

total = neg + pos

print('Total Examples are {}\n Positives are {:.2f}% of total\n Negatives are {:.2f}% of total'.format(total, 100*pos/total, 100*neg/total))
amount_val = raw_df['Amount'].values

fig = sns.distplot(amount_val)

fig.set_title('Distribution of Transection Amount')

fig.set_xlim([min(amount_val), max(amount_val)])
time_val = raw_df['Time'].values

fig = sns.distplot(time_val,  color = 'b')

fig.set_title('Distribution of Transection Time')

fig.set_xlim([min(time_val), max(time_val)])
# sklearn StandardScaler will set the mean to 0 and standard deviation to 1.  

# RobustScaler is less prone to outliers.

raw_df['scaled_amount'] = RobustScaler().fit_transform(raw_df['Amount'].values.reshape(-1,1))

raw_df['scaled_time'] = RobustScaler().fit_transform(raw_df['Time'].values.reshape(-1,1))



raw_df.drop(['Time','Amount'], axis = 1, inplace = True)
fig = sns.distplot(raw_df['scaled_time'].values)

fig.set_title('Distribution of Scaled Transection Time')
fig = sns.distplot(raw_df['scaled_amount'].values,  color = 'b')

fig.set_title('Distribution of Scaled Transection Amount')
pos
new_df = raw_df.copy()

new_df = new_df.sample(frac = 1)

new_pos = new_df.loc[new_df['Class'] == 1]

new_neg = new_df.loc[new_df['Class'] == 0][:492]

norm_dis_df = pd.concat([new_pos, new_neg])

norm_df = norm_dis_df.sample(frac = 1, random_state=42)

norm_df.head()
print('Distribution of the Classes in the subsample dataset')

print(norm_df['Class'].value_counts()/len(norm_df))

sns.countplot('Class', data = norm_df)
plt.figure(figsize=(12,10))

sns.heatmap(norm_df.corr(),cmap='coolwarm_r')
fig, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(x = 'Class', y = 'V1', data = norm_df, ax = axes[0])

axes[0].set_title('V1 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V3', data = norm_df, ax = axes[1])

axes[1].set_title('V3 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V5', data = norm_df, ax = axes[2])

axes[2].set_title('V5 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V6', data = norm_df, ax = axes[3])

axes[3].set_title('V6 Negative Correlation with Class')
fig, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(x = 'Class', y = 'V7', data = norm_df, ax = axes[0])

axes[0].set_title('V7 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V9', data = norm_df, ax = axes[1])

axes[1].set_title('V9 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V10', data = norm_df, ax = axes[2])

axes[2].set_title('V10 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V12', data = norm_df, ax = axes[3])

axes[3].set_title('V12 Negative Correlation with Class')
fig, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(x = 'Class', y = 'V14', data = norm_df, ax = axes[0])

axes[0].set_title('V14 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V16', data = norm_df, ax = axes[1])

axes[1].set_title('V16 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V17', data = norm_df, ax = axes[2])

axes[2].set_title('V17 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V18', data = norm_df, ax = axes[3])

axes[3].set_title('V18 Negative Correlation with Class')
v1_pos = norm_df['V1'].loc[norm_df['Class'] == 1].values

q25, q75 = np.percentile(v1_pos, 25), np.percentile(v1_pos, 75)



v1_iqr = q75 - q25



v1_lower, v1_upper = q25 - v1_iqr * 1.5, q75 + v1_iqr * 1.5



norm_df = norm_df.drop(norm_df[(norm_df['V1'] > v1_upper)|(norm_df['V1'] < v1_lower)].index)
v3_pos = norm_df['V3'].loc[norm_df['Class'] == 1].values

q25, q75 = np.percentile(v3_pos, 25), np.percentile(v3_pos, 75)



v3_iqr = q75 - q25



v3_lower, v3_upper = q25 - v3_iqr * 1.5, q75 + v3_iqr * 1.5



norm_df = norm_df.drop(norm_df[(norm_df['V3'] > v3_upper)|(norm_df['V3'] < v3_lower)].index)
v5_pos = norm_df['V5'].loc[norm_df['Class'] == 1].values

q25, q75 = np.percentile(v5_pos, 25), np.percentile(v5_pos, 75)



v5_iqr = q75 - q25



v5_lower, v5_upper = q25 - v5_iqr * 1.5, q75 + v5_iqr * 1.5



norm_df = norm_df.drop(norm_df[(norm_df['V5'] > v5_upper)|(norm_df['V5'] < v5_lower)].index)
v6_pos = norm_df['V6'].loc[norm_df['Class'] == 1].values

q25, q75 = np.percentile(v6_pos, 25), np.percentile(v6_pos, 75)



v6_iqr = q75 - q25



v6_lower, v6_upper = q25 - v6_iqr * 1.5, q75 + v6_iqr * 1.5



norm_df = norm_df.drop(norm_df[(norm_df['V6'] > v6_upper)|(norm_df['V6'] < v6_lower)].index)
v7_pos = norm_df['V7'].loc[norm_df['Class'] == 1].values

q25, q75 = np.percentile(v7_pos, 25), np.percentile(v7_pos, 75)



v7_iqr = q75 - q25



v7_lower, v7_upper = q25 - v7_iqr * 1.5, q75 + v7_iqr * 1.5



norm_df = norm_df.drop(norm_df[(norm_df['V7'] > v7_upper)|(norm_df['V7'] < v7_lower)].index)
v9_pos = norm_df['V9'].loc[norm_df['Class'] == 1].values

q25, q75 = np.percentile(v9_pos, 25), np.percentile(v9_pos, 75)



v9_iqr = q75 - q25



v9_lower, v9_upper = q25 - v9_iqr * 1.5, q75 + v9_iqr * 1.5



norm_df = norm_df.drop(norm_df[(norm_df['V9'] > v9_upper)|(norm_df['V9'] < v9_lower)].index)
v10_pos = norm_df['V10'].loc[norm_df['Class'] == 1].values

q25, q75 = np.percentile(v10_pos, 25), np.percentile(v10_pos, 75)



v10_iqr = q75 - q25



v10_lower, v10_upper = q25 - v10_iqr * 1.5, q75 + v10_iqr * 1.5



norm_df = norm_df.drop(norm_df[(norm_df['V10'] > v10_upper)|(norm_df['V10'] < v10_lower)].index)
v12_pos = norm_df['V12'].loc[norm_df['Class'] == 1].values

q25, q75 = np.percentile(v12_pos, 25), np.percentile(v12_pos, 75)



v12_iqr = q75 - q25



v12_lower, v12_upper = q25 - v12_iqr * 1.5, q75 + v12_iqr * 1.5



norm_df = norm_df.drop(norm_df[(norm_df['V12'] > v12_upper)|(norm_df['V12'] < v12_lower)].index)
v14_pos = norm_df['V14'].loc[norm_df['Class'] == 1].values

q25, q75 = np.percentile(v14_pos, 25), np.percentile(v14_pos, 75)



v14_iqr = q75 - q25



v14_lower, v14_upper = q25 - v14_iqr * 1.5, q75 + v14_iqr * 1.5



norm_df = norm_df.drop(norm_df[(norm_df['V14'] > v14_upper)|(norm_df['V14'] < v14_lower)].index)
fig, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(x = 'Class', y = 'V1', data = norm_df, ax = axes[0])

axes[0].set_title('V1 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V3', data = norm_df, ax = axes[1])

axes[1].set_title('V3 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V5', data = norm_df, ax = axes[2])

axes[2].set_title('V5 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V6', data = norm_df, ax = axes[3])

axes[3].set_title('V6 Negative Correlation with Class')
fig, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(x = 'Class', y = 'V7', data = norm_df, ax = axes[0])

axes[0].set_title('V7 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V9', data = norm_df, ax = axes[1])

axes[1].set_title('V9 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V10', data = norm_df, ax = axes[2])

axes[2].set_title('V10 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V12', data = norm_df, ax = axes[3])

axes[3].set_title('V12 Negative Correlation with Class')
fig, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(x = 'Class', y = 'V14', data = norm_df, ax = axes[0])

axes[0].set_title('V14 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V16', data = norm_df, ax = axes[1])

axes[1].set_title('V16 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V17', data = norm_df, ax = axes[2])

axes[2].set_title('V17 Negative Correlation with Class')



sns.boxplot(x = 'Class', y = 'V18', data = norm_df, ax = axes[3])

axes[3].set_title('V18 Negative Correlation with Class')
# Use a utility from sklearn to split and shuffle the dataset

labels = norm_df['Class']

features = norm_df.drop('Class', axis = 1)

train_feature, test_feature, train_label, test_label = train_test_split(features, labels, test_size = 0.2, random_state=42)

train_feature, val_feature, train_label, val_label = train_test_split(features, labels, test_size = 0.2)



train_label = train_label.values

test_label = test_label.values

val_label = val_label.values

train_feature = train_feature.values

test_feature = test_feature.values

val_feature = val_feature.values
from keras.utils import to_categorical



train_label = to_categorical(train_label)

test_label = to_categorical(test_label)

val_label = to_categorical(val_label)
print('training label shape:', train_label.shape)

print('training feature shape', train_feature.shape)

print('testing label shape', test_label.shape)

print('testing feature shape', test_feature.shape)

print('validation label shape', val_label.shape)

print('validation feature shape', val_feature.shape)
model = Sequential()



model.add(Dense(input_dim = train_feature.shape[-1], units = 200, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(units = 200, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(units = 2, activation = 'sigmoid'))



model.summary()



model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x = train_feature, y = train_label, batch_size = 30, epochs = 30)
train_result = model.evaluate(x = train_feature, y = train_label)

validate_result = model.evaluate(x = val_feature, y = val_label)
predict_result = model.evaluate(x = test_feature, y = test_label)