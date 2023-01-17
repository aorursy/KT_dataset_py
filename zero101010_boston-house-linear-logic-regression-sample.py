# import 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# read dataset

dataset = pd.read_csv('../input/housing.csv')

dataset.shape
# drop values not a number

dataset = dataset.dropna()

display(dataset.shape)

# get a random values of dataframe

display(dataset.sample(5))
# find values in median_house_values that is equal 50000

dataset.loc[dataset["median_house_value"] ==500001].count()
# delete the median_house_values that is equal 50000

dataset = dataset.drop(dataset.loc[dataset["median_house_value"] ==500001].index)

dataset.shape
# type of clasifications that exist in ocean_aproximity

dataset["ocean_proximity"].unique()
#Convert string values in numeric values using one-hot encoding, and transfrom each string value in a new column

housing_data = pd.get_dummies(dataset, columns = ['ocean_proximity'])

housing_data.head()
# split features(X) and target_values(Y)

X = housing_data.drop("median_house_value",axis=1)

Y = housing_data["median_house_value"]

display(X.head(5))

display(Y.head(5))
# split train and test

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

display(X_train.shape)

display(X_test.shape)
# import the model LinearRegression

from sklearn.linear_model import LinearRegression

# normalize= true mean that all the features will be in range 0 to 1 to improve the performance 

model = LinearRegression(normalize=True)

model.fit(X_train,Y_train)

# score of the model

display("Trainig score {:.2f}%".format(model.score(X_train,Y_train)*100))

predictors = X_train.columns

predictors
# find what features impact more in the the result of price

coef = pd.Series(model.coef_,predictors).sort_values()

coef
# predict using the model

predict = model.predict(X_test)

predict_actual = pd.DataFrame({"predicted":predict,"actual":Y_test})

predict_actual.head(10)
# Score of model

from sklearn.metrics import r2_score

print('testing_score:',r2_score(Y_test,predict))
fig, ax = plt.subplots(figsize=(12,8))

plt.scatter(Y_test,predict)

plt.show()