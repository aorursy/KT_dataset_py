import pandas as pd

from sklearn.impute import SimpleImputer

import numpy as np

from sklearn import preprocessing

from sklearn import linear_model
train_data = pd.read_csv('train.csv') #1460 Data instances

test_data = pd.read_csv('test.csv') #1459 Data instances

train_data.head()
# Concatenate the train and test data to make pre-processing on both



train_data_y = train_data[['SalePrice']]



# Selected by Intution only

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr','OverallQual','GarageCars',

           'GarageArea','PoolArea']

train_data_x = train_data[features]

test_data_x = test_data[features]





#train_test_data = [train_data_x,test_data_x]





# Check NAN for train



nan_sizes_inputs = []



for item in features:

    nan_sizes_inputs.append(train_data_x[item].isnull().sum())

    

for size in nan_sizes_inputs:

    print(size)

        
# Check NAN for test





nan_sizes_inputs = []



for item in features:

    nan_sizes_inputs.append(test_data_x[item].isnull().sum())

    

for size in nan_sizes_inputs:

    print(size)

#Impute with the mean in the test set

test_data_x['GarageArea'] = test_data_x['GarageArea'].fillna(test_data_x['GarageArea'].mean())

test_data_x['GarageCars'] = test_data_x['GarageCars'].fillna(test_data_x['GarageCars'].mean())
# Re-Check NAN for test





nan_sizes_inputs = []



for item in features:

    nan_sizes_inputs.append(test_data_x[item].isnull().sum())

    

for size in nan_sizes_inputs:

    print(size)

# check data type for every attribute

test_data_x.dtypes



print(test_data_x)
# Normalization of the data

train_data_x = (train_data_x - train_data_x.mean() ) / train_data_x.std()

test_data_x = (test_data_x - test_data_x.mean() ) / test_data_x.std()
model = linear_model.LinearRegression()

model.fit(train_data_x, train_data_y)



test_data_y = (model.predict(test_data_x)).flatten()



print(test_data_y)


#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'Id':test_data['Id'],'SalePrice':test_data_y})



#Visualize the first 5 rows

submission.head()



submission.to_csv('Houses_output_predictions.csv',index=False)