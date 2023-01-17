# library we'll need

import pandas as pd



# read in all three datasets (you'll pick one to use later)

recpies = pd.read_csv("../input/epirecipes/epi_r.csv")

bikes = pd.read_csv("../input/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv")

weather = pd.read_csv("../input/szeged-weather/weatherHistory.csv")
recpies.dropna(inplace=True)

recpies = recpies[recpies['calories']<10000]
# are the ratings all numeric?

print("Is this variable numeric?")

recpies['rating'].dtype == 'float64'
# are the ratings all integers?

print("Is this variable only integers?")

recpies['rating'].dtype == 'int'
import matplotlib.pyplot as plt



plt.scatter(recpies['calories'], recpies['dessert'])

plt.xlabel('calories')

plt.ylabel('dessert')

plt.show()
from sklearn import linear_model

import numpy as np

logreg = linear_model.LogisticRegression()

X = recpies['calories'][:, np.newaxis]

y = recpies['dessert']

logreg.fit(X, y)



# and plot the result

plt.scatter(X.ravel(), y, color='black', zorder=20)

plt.plot(X, logreg.predict_proba(X)[:,1], color='blue', linewidth=3)

plt.xlabel('calories')

plt.ylabel('dessert')



plt.show()
bikes.head()
x = bikes['High Temp (°F)'][:, np.newaxis]

y = bikes['Total']

plt.scatter(x,y)
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x,y)

y_pred = reg.predict(x)

plt.scatter(x, y)

plt.plot(x, y_pred, color='blue', linewidth=3)