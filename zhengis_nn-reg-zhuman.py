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
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error 

from matplotlib import pyplot as plt

import seaborn as sb

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import warnings 

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)

from xgboost import XGBRegressor

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') 

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
target = train.SalePrice

train.drop(['SalePrice'],axis = 1 , inplace = True)



combined = train.append(test)

combined.reset_index(inplace=True)

combined.drop(['index', 'Id'], inplace=True, axis=1)
combined.describe()
def get_cols_with_no_nans(df,col_type):

    '''

    Arguments :

    df : The dataframe to process

    col_type : 

          num : to only get numerical columns with no nans

          no_num : to only get nun-numerical columns with no nans

          all : to get any columns with no nans    

    '''

    if (col_type == 'num'):

        predictors = df.select_dtypes(exclude=['object'])

    elif (col_type == 'no_num'):

        predictors = df.select_dtypes(include=['object'])

    elif (col_type == 'all'):

        predictors = df

    else :

        print('Error : choose a type (num, no_num, all)')

        return 0

    cols_with_no_nans = []

    for col in predictors.columns:

        if not df[col].isnull().any():

            cols_with_no_nans.append(col)

    return cols_with_no_nans
num_cols = get_cols_with_no_nans(combined , 'num')

cat_cols = get_cols_with_no_nans(combined , 'no_num')
print ('Number of numerical columns with no nan values :',len(num_cols))

print ('Number of nun-numerical columns with no nan values :',len(cat_cols))
combined = combined[num_cols + cat_cols]

combined.hist(figsize = (12,10))

plt.show()
train = train[num_cols + cat_cols]

train['Target'] = target



C_mat = train.corr()

fig = plt.figure(figsize = (15,15))



sb.heatmap(C_mat, vmax = .8, square = True)

plt.show()
def oneHotEncode(df,colNames):

    for col in colNames:

        if( df[col].dtype == np.dtype('object')):

            dummies = pd.get_dummies(df[col],prefix=col)

            df = pd.concat([df,dummies],axis=1)



            #drop the encoded column

            df.drop([col],axis = 1 , inplace=True)

    return df

    



print('There were {} columns before encoding categorical features'.format(combined.shape[1]))

combined = oneHotEncode(combined, cat_cols)

print('There are {} columns after encoding categorical features'.format(combined.shape[1]))
def split_combined():

    global combined

    train = combined[:1460]

    test = combined[1460:]



    return train , test 

  

train_data, test_data = split_combined()
NN_model = Sequential()



# The Input Layer :

NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train_data.shape[1], activation='relu'))



# The Hidden Layers :

NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))



# The Output Layer :

NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))



# Compile the network :

NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

NN_model.summary()
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 

checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

callbacks_list = [checkpoint]
NN_model.fit(train_data, target, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)
# Load wights file of the best model :

# wights_file = 'Weights-397--18993.77574.hdf5' # choose the best checkpoint 

# NN_model.load_weights(wights_file) # load it

# NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
def make_submission(prediction, sub_name):

  my_submission = pd.DataFrame({'Id':test.Id,'SalePrice':prediction})

  my_submission.to_csv('{}.csv'.format(sub_name),index=False)

  print('A submission file has been made')



predictions = NN_model.predict(test_data)

make_submission(predictions[:,0],'submission(NN).csv')
train_X, val_X, train_y, val_y = train_test_split(train_data, target, test_size = 0.25, random_state = 14)
model = RandomForestRegressor()

model.fit(train_X,train_y)



# Get the mean absolute error on the validation data

predicted_prices = model.predict(val_X)

MAE = mean_absolute_error(val_y , predicted_prices)

print('Random forest validation MAE = ', MAE)
predicted_prices = model.predict(test_data)

make_submission(predicted_prices,'Submission(RF).csv')
XGBModel = XGBRegressor()

XGBModel.fit(train_X,train_y , verbose=False)



# Get the mean absolute error on the validation data :

XGBpredictions = XGBModel.predict(val_X)

MAE = mean_absolute_error(val_y , XGBpredictions)

print('XGBoost validation MAE = ',MAE)


XGBpredictions = XGBModel.predict(test_data)

make_submission(XGBpredictions,'Submission(XGB).csv')
from lightgbm import LGBMRegressor

LGBM = LGBMRegressor(n_estimators = 1000)

LGBM.fit(train_X,train_y)
LGBMpredictions = LGBM.predict(val_X)

MAE = mean_absolute_error(val_y , LGBMpredictions)

print('LGBM validation MAE = ',MAE)


LGBMpredictions = LGBM.predict(test_data)

make_submission(LGBMpredictions,'Submission(LGBM).csv')