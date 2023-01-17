# # library we'll need
# library(tidyverse)

# # read in all three datasets (you'll pick one to use later)
# recpies <- read_csv("../input/epirecipes/epi_r.csv")
# bikes <- read_csv("../input/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv")
# weather <- read_csv("../input/szeged-weather/weatherHistory.csv")

#Python Code
import pandas as pd

recipes = pd.read_csv("../input/epirecipes/epi_r.csv")
bikes = pd.read_csv("../input/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv")
weather = pd.read_csv("../input/szeged-weather/weatherHistory.csv")
# # quickly clean our dataset
# recpies <- recpies %>%
#     filter(calories < 10000) %>% # remove outliers
#     na.omit() # remove rows with NA values

#Python Code

recipes.dropna(inplace=True)
recipes = recipes[recipes['calories'] < 10000]
# # are the ratings all numeric?
# print("Is this variable numeric?")
# is.numeric(recpies$rating)

#Python Code

print("Is this variable numeric?")
recipes['rating'].dtype == 'float64'
# # are the ratings all integers?
# print("Is this variable only integers?")
# all.equal(recpies$rating, as.integer(recpies$rating)) == T

#Python Code
print("Is this variable only integers?")
recipes['rating'].dtype == "int"
# # plot calories by whether or not it's a dessert
# ggplot(recpies, aes(x = calories, y = dessert)) + # draw a 
#     geom_point()  # add points

#Python Code

import matplotlib.pyplot as plt

plt.scatter(recipes['calories'],recipes['dessert'])
plt.xlabel('calories')
plt.ylabel('dessert')
plt.show()
# # plot & add a regression line
# ggplot(recpies, aes(x = calories, y = dessert)) + # draw a 
#     geom_point() + # add points
#     geom_smooth(method = "glm", # plot a regression...
#     method.args = list(family = "binomial")) # ...from the binomial family

#Python Code
from sklearn import linear_model
import numpy as np

log_reg = linear_model.LogisticRegression()
X = recipes['calories'][:, np.newaxis] #Adding a dimension with np.newaxis
y = recipes['dessert']
log_reg.fit(X,y)

#and plot the result
plt.scatter(X.ravel(), y, color = 'black', zorder = 20)
plt.plot(X, log_reg.predict_proba(X)[:,1], color = 'blue', linewidth = 3)
plt.xlabel('calories')
plt.ylabel('dessert')
plt.show()
# your work goes here! :)

bikes.head()
X = bikes['High Temp (°F)'][:,np.newaxis]
y = bikes['Total']
plt.scatter(X,y)
print("The type of Total variable")
bikes['Total'].dtype == 'int'
linear_reg = linear_model.LinearRegression()
linear_reg.fit(X,y)
y_pred = linear_reg.predict(X)
plt.scatter(X,y)
plt.plot(X, y_pred, color = 'blue', linewidth = 3)