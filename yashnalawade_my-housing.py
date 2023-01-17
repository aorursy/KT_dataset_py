import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor 

from sklearn.metrics import r2_score

import seaborn as sns

import csv

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
n_features = 15

corr_mat = train_data.corr()

top_features = list(corr_mat.nlargest(n_features, 'SalePrice')['SalePrice'].index)

#top_cor = train_data[top_features].corr()

#sns.heatmap(corr_mat)

#print(top_features)

data=train_data[top_features]

data = data[top_features].fillna(method='ffill')
split_index = int(len(data)*0.7)

y_train = np.array(data['SalePrice'].iloc[0:split_index])

# y_train = y_train.reshape([len(y_train), 1])



y_test = np.array(data['SalePrice'].iloc[split_index:])

# y_test = y_test.reshape([len(y_test), 1])



if 'SalePrice' in top_features:

    top_features.remove('SalePrice')



# training and testing input

x_train = np.array(data[top_features][0:split_index])



x_test = np.array(data[top_features][split_index:])



print('training input size: ', x_train.shape)

print('training output size: ', y_train.shape)

print('\ntesting input size: ', x_test.shape)

print('testing output size: ', y_test.shape)
model = RandomForestRegressor()

model.fit(x_train, y_train)



train_prediction = model.predict(x_train)

test_prediction = model.predict(x_test)



print('R2_train: ', r2_score(y_train, train_prediction))

print('R2_test: ',  r2_score(y_test, test_prediction))
prediction_data = test_data.fillna(method='ffill')

output = model.predict(prediction_data[top_features])

pred_list = [['Id', 'SalePrice']]

for i in range(len(prediction_data)):

    pred_list.append([prediction_data['Id'].iloc[i], output[i]])

    

with open("my_submission.csv","w+") as my_csv:

    csvWriter = csv.writer(my_csv,delimiter=',')

    csvWriter.writerows(pred_list)

print("Your submission was successfully saved!")