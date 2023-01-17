# numpy for Math Calculations 

import numpy as np

# random for make some random numbers that we are going to use next

import random

# mathplotlib for making some visualisations

import matplotlib.pyplot as plt



# and linear_model from sklearn 

from sklearn import linear_model
x = np.linspace(0,100,500)

x = np.array([i+random.randint(-1,1) for i in x])
y = np.array([i+random.randint(-10,10) for i in x])
## the next line is not important it's just to change the default size of our plot and make it 10*10

plt.rcParams['figure.figsize'] = 10, 10



###plot x and y

plt.plot(x,y,'x')
from sklearn.model_selection import train_test_split
# this is how we use the function, the random state is just a random number, you can pass any number,we will take 

# a 40% of data as a test data, (in the realy life problems the test data is only 20% to 25% , but it's not a rule 

# it depends on the size of the dataset)

X_train, X_test, y_train, y_test = train_test_split(x, y,random_state=42,test_size=0.4)
# let's plot the data 

# we will show the train data in the X shape with the red color 

plt.plot(X_train,y_train,'rx',label = 'train data')

# we will use the blue color to display the test data

plt.plot(X_test,y_test,'bo',label = 'test data')

# the next data is just to show the labels so any one can read and understand the plot

plt.legend()
# to use the linear regression algorithm we only need to import it like this

from sklearn.linear_model import LinearRegression

# so now all what we need to do is to train the algo, but before that we have to be sure that our X_train

# is a 2d array

X_train.ndim
X_train = X_train.reshape(-1,1)

X_train.ndim
linear = LinearRegression().fit(X_train, y_train)
# reshape X_test

X_test = X_test.reshape(-1,1)

# calculate the score

linear.score(X_test.reshape(-1,1),y_test)
linear.coef_
# so there is no b and that's good because like you know the start of our data was 0 so no b 

a = linear.coef_[0]
# to draw the line we will take the fitst and the last points 

x_line = np.array([[0],[100]])

y_line = np.array([[0],[a*100]])
plt.plot(x,y,'x')

plt.plot(x_line,y_line,'-')