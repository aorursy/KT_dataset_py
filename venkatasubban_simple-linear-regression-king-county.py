# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#!pip install uszipcode



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import regular packages

#Step to import the packages

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

#Read the Input file and go through the data.

df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')

df.head()
#lets check any null values are there. 

df.isnull().count()

# NO null values are found. 
# As we are supposed to predict the house price, there should be strong correlation between Square feet

# and house price. Lets try summing up square feet price and try to plot it against price to get a view over it



# Let us sum all the square feets and try to correlate against Price. 



df['sum'] = df['sqft_living']+df['sqft_above']+df['sqft_basement']

y = df['price']

x=df['sum']

plt.scatter(x,y)

plt.xlabel('Sqft Sum')

plt.ylabel('Price in  $100k')
# The above graph Sqft Sum vs Price seems to be linearly correlated. We can see strong corelationship. 

# Lets check on other possible way to visualize the data. 

# Going through the data, zipcode plays an important role as prices of certian codes (downtown/Near commerical areas)

# may have higher rates. 
### let us find the top 10 count of houses available based on zipcode



k=df[['zipcode','id']].groupby(['zipcode']).count().sort_values(by='id',ascending=False).head(10).plot.bar()

plt.figure(figsize=(40,5))

plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False

plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True



# set individual bar labels using above list

for i in k.patches:

    # get_x pulls left or right; get_height pushes up or down

    k.text(i.get_x()-.03, i.get_height()+.75, \

            str(round((i.get_height()), 2)))

# Let us find Per Sqft Price based on total sum arrived for indivudal houses.  

df['Per_Sqft'] =  (df['price']/df['sum'])

# Let us correlate per sqft price against total sum



y=df['Per_Sqft']

x=df['sum']

plt.scatter(x,y)



#on similar note, we will compare it against other parameters as well 

x=df['sqft_living']

plt.scatter(x,y,color='green')

#However the above model is not right as Per_sqft price vary from house to house depending on the size of the house 

#and facilities it might have. Lets try to correlate the persqft averaged out against the zipcode. 

#Lets group zipcodes and find out average sqft price. 



zipgrp=pd.DataFrame(df.groupby('zipcode').mean())

zipgrp.sort_values(by=['Per_Sqft'],inplace=True, ascending=False)

zipgrp.reset_index(level=0, inplace=True)

 

# Moving top 10 data for graphical representation     

zipgrp1 = zipgrp.head(10)

zipgrp1.filter(['zipcode','Per_Sqft']).head(5)



# the top zipcodes which has highest sellig prices are below. Please scroll right to see the 
# find out locations with highest selling units with high per square feet price

plt.figure(figsize=(35,5))

sns.set(style="whitegrid")

plot_order = zipgrp1.sort_values(by='Per_Sqft', ascending=False).zipcode.values

g=sns.barplot(x=df['zipcode'],y=df['Per_Sqft'],data=df,order=plot_order)

plt.ylabel('Per_Sqft in ($)',fontsize=18)

plt.xlabel('Zipcode',fontsize=18)

for i in g.patches:

    # get_x pulls left or right; get_height pushes up or down

    g.text(i.get_x()+.3, i.get_height()+1, \

            str(round((i.get_height()), 2)))
# Let us try to do few correlations as well. A strong correlation against 'Price' should help us to understand

# how other variables are related to it.

correlation_matrix = df.corr()

plt.figure(figsize = (10,10))

s = correlation_matrix['price'].sort_values(ascending = False)

print(s)

s.plot.bar()

plt.xlabel('Zipcode',fontsize=18)

plt.ylabel('Correlation factor',fontsize=18)

#for i in s.patches:

#    # get_x pulls left or right; get_height pushes up or down

#    s.text(i.get_x()+.3, i.get_height()+1, \

#            str(round((i.get_height()), 2)))
# Splitting training and test data 

# Considered 'Per_Sqft'data as well as it improves covariance to a greater extent.

y = df['price']

X = df[['sqft_living','grade','sqft_above','sqft_living15','bathrooms','view','bedrooms','lat','floors','waterfront',

        'sqft_basement','sqft_lot','yr_renovated','yr_built','condition','long','Per_Sqft']]

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 

                                                    random_state=1) 
# importing linear regressionmodel and predicting output



X_test.count()

from sklearn.linear_model import LinearRegression

reg = LinearRegression() 

reg.fit(X_train, y_train)

prediction = reg.predict(X_test)

print('Prediction', prediction,sep='\n')
#To retrieve the intercept:

print('Intercept', reg.intercept_)

#For retrieving the slope:

print('Regression Coefficients: \n', reg.coef_) 

# variance score: 1 means perfect prediction 

print('Variance score: {}'.format(reg.score(X_test, y_test)))
from sklearn import metrics

print('Mean Absolute error',metrics.mean_absolute_error(y_test,prediction))

print('Mean Squared Error',metrics.mean_squared_error(y_test,prediction))

print('Rootmean squared Error',np.sqrt(metrics.mean_squared_error(y_test,prediction)))

plt.scatter(y_test, prediction)

plt.xlabel("Prices: $Y_i$")

plt.ylabel("Predicted prices: $\hat{Y}_i$")

plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

plt.show()
# Let us check on residuals

df2= X_test

df1 = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})

df1['Residuals'] = df1['Actual'] - df1['Predicted']

df1.head(20)
# Lets see residuals are distributed randomly or does it follow a pattern. 

# let us try to plot a graph between predicted values and residuals



x=df1['Predicted']

y=df1['Residuals']



plt.scatter(x,y,color='blue')



plt.xlabel("Predicted")

plt.ylabel("Residuals")

plt.figure(figsize=(100,10))

plt.grid(True)



#Residuals were not exactly randomly distributed due to mulitcolinearity factors. Need to transform the data with diffrent

# regrression models to match the best fit. Will keep posted the second version on this. Few more residual visualization for reference. 
from yellowbrick.regressor import residuals_plot

viz = residuals_plot(LinearRegression(), X_train, y_train, X_test, y_test)
from yellowbrick.regressor import ResidualsPlot

from sklearn.linear_model import Ridge

model = ResidualsPlot(Ridge())

model.fit(X_train, y_train)

model.score(X_test, y_test)

model.show()