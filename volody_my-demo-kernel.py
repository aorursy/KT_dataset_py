import numpy as np  # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

                                 

import warnings     # `do not disturbe` mode

warnings.filterwarnings('ignore')
forest_train = pd.read_csv('../input/learn-together/train.csv')

# print a summary of the data

forest_train.describe()
forest_train.isnull().values.any()
forest_train.columns
y = forest_train.Cover_Type
forest_features = ['Elevation', 'Aspect', 

                   'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 

                   'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']



X = forest_train[forest_features]



X.describe()
X.head()
from sklearn.tree import DecisionTreeRegressor



# Define model. Specify a number for random_state to ensure same results each run

forest_model = DecisionTreeRegressor(random_state=1)



# Fit model

forest_model.fit(X, y)
print("Making predictions for the following 5 trees:")

print(X.head())

print("The predictions are")

print(forest_model.predict(X.head()))
from sklearn.metrics import mean_absolute_error



predicted_cover_type = forest_model.predict(X)

mean_absolute_error(y, predicted_cover_type)
from sklearn.model_selection import train_test_split



# split data into training and validation data, for both features and target

# The split is based on a random number generator. Supplying a numeric value to

# the random_state argument guarantees we get the same split every time we

# run this script.

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model

forest_model = DecisionTreeRegressor()

# Fit model

forest_model.fit(train_X, train_y)



# get predicted prices on validation data

val_predictions = forest_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



# compare MAE with differing values of max_leaf_nodes

for max_leaf_nodes in [5, 50, 200, 500]:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %.2f" %(max_leaf_nodes, my_mae))
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



r_forest_model = RandomForestRegressor(n_estimators=1100,

                                       max_features=4,

                                       random_state=70,

                                       max_depth=120,

                                       bootstrap=True)



r_forest_model.fit(train_X, train_y)

r_forest_preds = r_forest_model.predict(val_X)

print(mean_absolute_error(val_y, r_forest_preds))
# path to file you will use for predictions

test_data_path = '../input/learn-together/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[forest_features]



# make predictions which we will submit. 

test_preds = np.round(r_forest_model.predict(test_X)).astype(int)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'Cover_Type': test_preds})

output.to_csv('submission.csv', index=False)