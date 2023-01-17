# Code you have previously used to load data

import pandas as pd

from sklearn.tree import DecisionTreeRegressor



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[feature_columns]



# Specify Model

iowa_model = DecisionTreeRegressor()

# Fit Model

iowa_model.fit(X, y)



print("First in-sample predictions:", iowa_model.predict(X.head()))

print("Actual target values for those homes:", y.head().tolist())



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex4 import *

print("Setup Complete")
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

step_1.check()
# The lines below will show you a hint or the solution.

# step_1.hint() 

# step_1.solution()

import pandas as pd

from sklearn.tree import DecisionTreeRegressor





# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)





step_2.check()
# step_2.hint()

# step_2.solution()
# Predict with all validation observations

val_predictions = iowa_model.predict(val_X )



step_3.check()
# step_3.hint()

# step_3.solution()
# print the top few validation predictions

print(val_predictions[1:6])    #printing top 5 values

# print the top few actual prices from validation data

print(val_y[1:6])

from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y , val_predictions)





print(val_mae)

step_4.check()
# step_4.hint()

# step_4.solution()