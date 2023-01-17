#======================================================================
# Data Workload - Data Preparation and Transformation  
#======================================================================
# Data engineer will be loading, manipulating, transforming, analyzing datasets.
# Need lots of memory, otherwise need to process in batches or in stages, which will take more time.
import pandas as pd
iris = pd.read_csv('../input/demodata/iris.csv')
print(iris[:5])
print(iris[-5:])
#
# visualization is very useful for "human" data scientist to decide what features available and that should be included in the ML model.
#
%matplotlib inline
import seaborn as sns; sns.set()
sns.pairplot(iris, hue='species', height=2.5);
#============================================================================================
# For this exercise, we will generate the training dataset based on a simple linear equation.
# We'll use linear regression model to be trained on this training data.  
#
# Generate training data for simple linear regression:
#   y = wt * x + bias    // Use a linear equation as the base, with a fixed weight and bias.
#   Add random distortion to the generated data data set.
#   Add intermitten spike to simulate "dirty" data.
#
#   x is the input data
#   y is the expected result
#============================================================================================
import matplotlib.pyplot as plt 
import numpy as np
import time

# Use same random seeds to generate same data set
rdm1 = np.random.RandomState(123)
rdm2 = np.random.RandomState(100)
#rdm2 = np.random.RandomState(int(time.time()))

# Set the weight and bias for the linear equation. The trained model should result in 
# weight and bias that is close to these values.
wt = 3
bias = 10

#
# Set the flag to generate spike/dirty data.  Data cleansing can be good or bad depending on the context. 
#  
add_spike = 1

#
# Generate the dataset for use in linear regression ML. 
#  
xd = []
yd = []
yc = []
for x in range(50):
   xd.append(x)
   y = (wt*x + bias)
   random_distortion = (25 - rdm1.randint(1,51))/2
   if ((x % 10) == 0):
     random_noise = rdm2.randint(50,100) * add_spike
   else:
     random_noise = 0
   yd.append(y + random_distortion + random_noise)
   yc.append(y)

xs = np.array(xd, dtype=float).reshape((-1, 1))
ys = np.array(yd, dtype=float).reshape((-1, 1))

# Visualize the sample data set
plt.figure(figsize=(10,10))
plt.scatter(xs,ys,color='red',label='data')
plt.plot(xd,yc,color='black',label = 'calc')
plt.legend()
plt.show()
#======================================================================
# Simple Linear Regression With scikit-learn
#======================================================================
#
# We will be using linear regression to plot/predict the value of y based on the value of x.
# Essentially, we will use ML to build a model function, f(x)=y.
#
from sklearn.linear_model import LinearRegression

#======================================================================
# Calculate best fit using linear reqgression
#======================================================================
model = LinearRegression()
model.fit(xs, ys)

r_sq = model.score(xs, ys)
print('coefficient of determination:', r_sq)
print('slope/weight:', model.coef_)
print('intercept/bias:', model.intercept_)
coef = model.coef_[0]

yp = []
for x in range(50):
   yp.append((coef*x + model.intercept_))

# Plot the sample data set and predicted data set
plt.figure(figsize=(10,10))
plt.scatter(xs,ys,color='red',label='data')
plt.plot(xd,yc,color='black',label = 'calc')
plt.plot(xd,yp,color='green',label = 'pred')
plt.legend()
plt.show()

#============================================================================
# Test the model with the trained model and with the trained weight and bias.
#============================================================================
x = 45
xs = np.array([x], dtype=float).reshape((-1, 1))
print("Input=", x, " Calculated =",wt*x+bias)
print("Predicted with Model", model.predict(xs))
print("Predicted with Weight and Bias", model.coef_*x+model.intercept_)

!rm *.pkl -f
!ls -lah

#============================================================================
# Save the trained model
#============================================================================
import pickle
import os

# Save to file in the current working directory
pkl_filename = "mymodel.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# List the uploaded files
import os
for dirname, _, filenames in os.walk('/kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#============================================================================
# Deploy the model from the pickle file.
# Instantiate a pre-trained model from the pickle file
#============================================================================

# Load from file the pre-trained model.
pkl_filename = "mymodel.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

# Test the model
#
x = 50
xs = np.array([x], dtype=float).reshape((-1, 1))
print("Input =", x, " Calculated =",wt*x+bias, "  Predicted", pickle_model.predict(xs))

!ls -lah