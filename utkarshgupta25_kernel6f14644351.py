import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics
import matplotlib.pyplot as plt

dataset = pd.read_json("../input/yelp.json", lines=True)

funny = []
cool = []
useful = []

dataset['funny'] = 0
dataset['cool'] = 0
dataset['useful'] = 0

dataset

for i in range(len(dataset['votes'])):
    useful.append(dataset['votes'][i]['useful'])
    funny.append(dataset['votes'][i]['funny'])
    cool.append(dataset['votes'][i]['cool'])
    
dataset.loc[:,['funny']] = funny
dataset.loc[:, ['cool']] = cool
dataset.loc[:, ['useful']] = useful

dataset.drop('votes', axis = 1, inplace = True)

print (dataset[['stars', 'funny', 'cool', 'useful']].corr())

x = dataset[['funny', 'cool', 'useful']]
y = dataset['stars'].values

regressor = lr()

x_train, x_test, y_train, y_test = tts(x,y)

regressor.fit(x_train,y_train)

predicted_values = regressor.predict(x_test)

mse = metrics.mean_squared_error(y_test, predicted_values)

print ("\n\nThe Root Mean Squared Error is: " + str(mse**(1.0/2.0)))

x = dataset[['funny', 'useful']]
y = dataset['stars'].values

print ("This time taken funny and useful as test cases as they are both inversely proprtional \nAnd no direct proportional relation is taken")

regressor = lr()

x_train, x_test, y_train, y_test = tts(x,y)

regressor.fit(x_train,y_train)

predicted_values = regressor.predict(x_test)

mse = metrics.mean_squared_error(y_test, predicted_values)

print ("\n\nThe Root Mean Squared Error is: " + str(mse**(1.0/2.0)))
x = dataset[['cool']]
y = dataset['stars'].values

print ("This time taken cool as test cases as it is direclty proportional")

regressor = lr()

x_train, x_test, y_train, y_test = tts(x,y)

regressor.fit(x_train,y_train)

predicted_values = regressor.predict(x_test)

mse = metrics.mean_squared_error(y_test, predicted_values)

print ("\n\nThe Root Mean Squared Error is: " + str(mse**(1.0/2.0)))
sns.regplot(dataset['funny'], dataset['stars'])
plt.show()
sns.regplot(dataset['cool'], dataset['stars'])
plt.show()
sns.regplot(dataset['useful'], dataset['stars'])
plt.show()