# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from learntools.core import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)

home_data.head(5)
home_data.info()
%matplotlib inline
import matplotlib.pyplot as plt
home_data.hist(bins=50, figsize=(30,20))
plt.show()
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(home_data, test_size = 0.2, random_state=42)
y_train = train_set['SalePrice'].copy()
y_test = test_set['SalePrice'].copy()

X_train = train_set.drop(['SalePrice'], axis='columns')
X_test = test_set.drop(['SalePrice'], axis='columns')
print("X_train columns: ", X_train.columns)
print("y_train: ", y_train.head(5))
home_data.corr()
home_data['MSSubClass'] = home_data['MSSubClass'].astype(str)
home_data['MSSubClass'].value_counts()
home_data.drop(['MSSubClass'], axis='columns', inplace=True)
home_data['MSZoning'].value_counts()
home_data.drop(['MSZoning'], axis='columns', inplace=True) # drop because some categories not represented
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

is_numeric_dtype(home_data['LotFrontage'])
home_data['LotFrontage'].isnull().sum()
home_data['LotFrontage'].fillna(0, inplace=True)

missing = (home_data.isnull().sum() / len(home_data))
missing[missing>0.5]

# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X,y)

# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
#test_data = test_data.dropna(axis=1)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features

test_X = test_data[features]

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)


# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)