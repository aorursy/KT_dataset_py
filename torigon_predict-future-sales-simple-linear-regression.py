import datetime

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



# Constants

DAT_PATH = '../input/'

OUT_PATH = '../output/'



# Loads data

train = pd.read_csv(DAT_PATH + 'sales_train.csv')

test = pd.read_csv(DAT_PATH + 'test.csv')



# Removes the data column

train_dat = train.drop('date', axis=1)



# Summarizes by month/shop/item

train_dat = train_dat.groupby(['date_block_num', 'shop_id', 'item_id'],).agg(

        {'item_price': 'mean', 'item_cnt_day': 'sum'})

train_dat.reset_index(inplace=True)



# Adds a month to the test data

test_dat = test.copy()

test_dat['date_block_num'] = 34
# Splits data

Xy_train = train_dat.loc[train_dat['date_block_num'] < 33, :]

Xy_test = train_dat.loc[train_dat['date_block_num'] == 33, :]

y_test_pred = []



#Xy_test = Xy_test.head(10)  # Comment this

print(datetime.datetime.now(), '*** Predict valid data START ***')

for index, row in Xy_test.iterrows():

#        row = Xy_test.iloc[0,:]

    one = Xy_train.loc[(Xy_train['shop_id'] == row['shop_id']) & (Xy_train['item_id'] == row['item_id']),

                    :].copy()



    if one.empty:

        # Gives zero if there was no previous record

        y_test_pred.append(0.0)

    else:

        # Predicts using linear regression

        model = LinearRegression()

        model.fit(one[['date_block_num']], one['item_cnt_day'])

        X_test = pd.DataFrame(

                {'date_block_num': [row['date_block_num']]})

        y_preds = model.predict(X_test)

        y_pred = y_preds[0]

        y_pred = 0 if y_pred < 0 else y_pred

        y_pred = 20 if y_pred > 20 else y_pred

        y_test_pred.append(y_pred)



print(datetime.datetime.now(), '*** Predict valid data END ***')



# Calculates RMSE

from sklearn.metrics import mean_squared_error

print('RMSE valid : %.3f' % \

      (np.sqrt(mean_squared_error(Xy_test['item_cnt_day'], y_test_pred))) )



# Plots observatio versus prediction

plt.figure()

plt.plot(Xy_test['item_cnt_day'], y_test_pred, 'o')

plt.show()
# Predicts using training data

Xy_train = train_dat.copy()

Xy_test = test_dat.copy()

y_test_pred = []



#Xy_test = Xy_test.head(10)  # Comment this for submission

print(datetime.datetime.now(), '*** Predict test data START ***')

for index, row in Xy_test.iterrows():

#    row = Xy_test.iloc[0,:]

    one = Xy_train.loc[(Xy_train['shop_id'] == row['shop_id']) & (Xy_train['item_id'] == row['item_id']),

                    :].copy()



    if one.empty:

        # Gives zero if there was no previous record

        y_test_pred.append(0.0)

    else:

        # Time series plot

#            one.plot(kind='line', x='date_block_num', y='item_cnt_day')



        # Predicts using linear regression

        model = LinearRegression()

        model.fit(one[['date_block_num']], one['item_cnt_day'])

        X_test = pd.DataFrame(

                {'date_block_num': [row['date_block_num']]})

        y_preds = model.predict(X_test)

        y_pred = y_preds[0]

        y_pred = 0 if y_pred < 0 else y_pred

        y_pred = 20 if y_pred > 20 else y_pred

        y_test_pred.append(y_pred)



print(datetime.datetime.now(), '*** Predict test data  END ***')



# Creates a CSV file to submit

#submit = pd.DataFrame({'ID': test_dat['ID'], 'item_cnt_month': y_test_pred})

#submit.to_csv(OUT_PATH + 'submit.csv', index=False)