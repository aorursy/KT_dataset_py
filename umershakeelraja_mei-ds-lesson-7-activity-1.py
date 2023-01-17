# import pandas
import pandas as pd

#import matplotlib for plotting
import matplotlib.pyplot as plt
# importing the data
body_data=pd.read_csv('../input/body-measurements/measures.csv')

# inspecting the dataset to check that it has imported correctly
body_data.head()
# describe and/or boxplots to check each field

body_data.plot.scatter(x='brozek', y='chest', figsize=(12,8))
# display a scatter diagram for age vs brozek body fat percentage
body_data.plot.scatter(x='age', y='brozek', figsize=(12,8))
plt.show()
# display more scatter diagrams
body_data.plot.scatter(x='age', y='brozek', figsize=(12,8))
plt.show()
body_data.plot.scatter(x='height', y='brozek', figsize=(12,8))
plt.show()
body_data.plot.scatter(x='weight', y='brozek', figsize=(12,8))
plt.show()
body_data.plot.scatter(x='chest', y='brozek', figsize=(12,8))
plt.show()
body_data.plot.scatter(x='abdom', y='brozek', figsize=(12,8))
plt.show()
# import numpy for handling lists/arrays
import numpy as np 

#import linear_model (for creating the model) and r2_score (for measuring how well the model fits the data)
from sklearn import linear_model
from sklearn.metrics import r2_score
# create the training subset from 80% of the original dataset
body_data_train=body_data.head(200)
body_data_train.head()
# create the testing subset from 20% of the original dataset
body_data_test=body_data.tail(50)
body_data_test.head()
# find the size of the testing and training datasets
print(np.shape(body_data_test), np.shape(body_data_train))
# create the target data for training as a list
target_train=body_data_train['brozek']
print(target_train)
# create an array for the input data
input_a_train=body_data_train[['weight']]

# define the model to be used as linear
model_a = linear_model.LinearRegression()

# fit a linear model to the data
model_a.fit(input_a_train, target_train)

# output the coefficients and y-intercept
print('Coefficients: \n', model_a.coef_)
print('Intercept: \n', model_a.intercept_)
# create the target data for testing as a list
target_test=body_data_test['brozek']
print(target_test)
# create list for the test input data
input_a_test=body_data_test[['weight']]

# use the input data to create a list of predictions
target_pred_a = model_a.predict(input_a_test)

# calculate and display the coefficient of determination
print("Brozek body fat percentage vs weight: R²="+str(r2_score(target_test, target_pred_a)))
# fit a linear model
target_train=body_data_train['brozek']
input_b_train=body_data_train[['abdom']]
model_b = linear_model.LinearRegression()
model_b.fit(input_b_train, target_train)
input_c_train=body_data_train[['chest']]
model_c = linear_model.LinearRegression()
model_c.fit(input_c_train, target_train)
print("Abdom:")
print('Coefficients: \n', model_b.coef_)
print('Intercept: \n', model_b.intercept_)
print("Chest:")
print('Coefficients: \n', model_c.coef_)
print('Intercept: \n', model_c.intercept_)

# calculate R²

input_b_test=body_data_test[['chest']]
target_pred_b = model_b.predict(input_b_test)
print("Brozek body fat percentage vs chest: R²="+str(r2_score(target_test, target_pred_b)))


input_c_test=body_data_test[['abdom']]
target_pred_c = model_c.predict(input_c_test)
print("Brozek body fat percentage vs abdom: R²="+str(r2_score(target_test, target_pred_c)))

# create an array for the input data
input_g_train=body_data_train[['weight','chest']]

# define the model to be used as linear
model_g = linear_model.LinearRegression()

# fit a linear model to the data
model_g.fit(input_g_train, target_train)

# output the coefficients and y-intercept
print('Coefficients: \n', model_g.coef_)
print('Intercept: \n', model_g.intercept_)
# create an array for the input data
input_g_test=body_data_test[['weight','chest']]

# use the input data to create a list of predictions
target_pred_g = model_g.predict(input_g_test)

# The coefficient of determination: 1 is perfect prediction
print("Brozek body fat percentage vs weight and chest size: R²="+str(r2_score(target_test, target_pred_g)))
# find the model
input_h_train = body_data_train[['chest', 'abdom']]
model_h=linear_model.LinearRegression()
model_h.fit(input_h_train, target_train)

input_i_train = body_data_train[['age', 'height']]
model_i=linear_model.LinearRegression()
model_i.fit(input_i_train, target_train)
# measure the model
print("Chest + abdom:")
print("Coefficients:"+str(model_h.coef_))
print("Intercept:"+str(model_h.intercept_))
print("Age + Height:")
print("Coefficients:"+str(model_i.coef_))
print("Intercept:"+str(model_i.intercept_))
input_h_test=body_data_test[['chest', 'abdom']]
target_pred_h=model_h.predict(input_h_test)
print("Brozek body fat percentage vs abdom and chest:R²="+str(r2_score(target_test, target_pred_h)))
input_i_test=body_data_test[['age', 'height']]
target_pred_i=model_i.predict(input_i_test)
print("Brozek body fat percentage vs age and height:R²="+str(r2_score(target_test, target_pred_i)))
# find the model


# measure the model