# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# lets import all the usual packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# Intention is here to find out dependent/ Independent variables which predict the car price that in turn helps Chinese

# car manfacturers to enter into USA market :) 

# We are doing this prediction during this uncertain times between USA and China.:)

# Stories apart, Lets read the data to our input variables

car_price = pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')

car_price1 = pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')
car_price.head()
#Lets have a look over the key paramaters.

car_price.describe()

# Below paramaters could be noticed

# could infer that Mean car price stands around $ 13 k

# peak rpm mean is around 5k

# horsepower stays around 104

# I am sure that number of doors, carbody, fueltype should also detremine the price

# let us check for any null data.

car_price.isnull().count()

# could see no null records are found. We have data in all the columns.
#Lets do few visualizations to understand the data. Lets concetrate over column types which are bound by categories and 

# find out how its related to price. This in turn will provide us a picture over factors deciding the car price



# Lets understand's basics. Fuel type and number of doors is very basic category we need to look out for. Lets see 

# how many cars distributed over fueltype and number of doors. 



Count_plot_Fueltype = sns.countplot(x='fueltype',data=car_price)

# set individual bar labels using above list

for i in Count_plot_Fueltype.patches:

    # get_x pulls left or right; get_height pushes up or down

    Count_plot_Fueltype.text(i.get_x()+.5, i.get_height()+.75, \

            str(round((i.get_height()), 2)))

plt.show()   

# Could infer from below graph, most of the cars do belong to gas and diesel. No electric cars :)

# On a lighter note, We are already seeing Tesla cars in the roads. 

# so this data is old :) May not be actually helpful for Chinese car makers :) if they are serious about futurisitc cars

# lets not jump out from focus, will proceed further to see what is there



Count_plot_doornumber = sns.countplot(x='doornumber',data=car_price)

# set individual bar labels using above list

for i in Count_plot_doornumber.patches:

    # get_x pulls left or right; get_height pushes up or down

    Count_plot_doornumber.text(i.get_x()+.5, i.get_height()+.75, \

            str(round((i.get_height()), 2)))
# Let us look column categories, find out how price ranges are distributed for those categories.

# This may give us understanding over factors that decide car price! 



ax = sns.boxplot(x='fueltype',y='price',hue='carbody',data=car_price)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          fancybox=True, shadow=True, ncol=5)



# could infer that hardtop and convertible cars have higher ranges. So Car body do have influence over car price
ax = sns.boxplot(x='fueltype',y='price',hue='doornumber',data=car_price)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          fancybox=True, shadow=True, ncol=5)

# No big variation could be seen with number of doors as we see mean price of gas cars with 2 doors and four doors

# almost wheel around with $10k
ax = sns.boxplot(x='fueltype',y='price',hue='fuelsystem',data=car_price)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          fancybox=True, shadow=True, ncol=5)

# fuel system with MPFI has higher price when compared to other fuel systems, it could be one of the deciding factors

# almost wheel around with $10k
ax = sns.boxplot(x='fueltype',y='price',hue='aspiration',data=car_price)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          fancybox=True, shadow=True, ncol=5)

# Aspiration system with turbo has got higher price
ax = sns.boxplot(x='fueltype',y='price',hue='enginetype',data=car_price)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          fancybox=True, shadow=True, ncol=5)

# The graph below clearly shows multiple engine types have different price ranges. 
ax = sns.boxplot(x='fueltype',y='price',hue='enginelocation',data=car_price)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          fancybox=True, shadow=True, ncol=5)

#Noticing an interesting observation, there are certain cars, with engine location at back is having hihger price

# around 35k. This could be kind of race cars!! We will find out about this cars later part
ax = sns.boxplot(x='fueltype',y='price',hue='drivewheel',data=car_price)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          fancybox=True, shadow=True, ncol=5)

# Price also varies with drivewheel category as well.
#Lets deep dive into few Technicalities of car, find out the relations. 

sns.jointplot(x='horsepower',y='price',data=car_price)

plt.show()

# there is a linear relationship between horsepower and price, more the horsepower, the price is getting increased. 

# However steep incresae in prices also can be noticed with increase in horsepower. 

# we will come  know to more about when we do correlation on later part of this. 



sns.jointplot(x='enginesize',y='price',data=car_price)

plt.show()

#engine size and car price again are again strongly correlated. Lets check on the correlation part for numerics. 
#As we lot of data hving text and adds a value to pricing, it is certainly need for  us to transform the data using

# label encoding. i have taken all this data. 

from sklearn import preprocessing

Encoding = preprocessing.LabelEncoder()



#Encoding = LabelEncoder()



car_price1['CarName'] = Encoding.fit_transform(car_price1['CarName'])

car_price1['fueltype'] = Encoding.fit_transform(car_price1['fueltype'])



car_price1['doornumber'] = Encoding.fit_transform(car_price1['doornumber'])

car_price1['aspiration']= Encoding.fit_transform(car_price1['aspiration'])





car_price1['fuelsystem'] = Encoding.fit_transform(car_price1['fuelsystem'])

car_price1['cylindernumber']=Encoding.fit_transform(car_price1['cylindernumber'])



car_price1['enginetype'] = Encoding.fit_transform(car_price1['enginetype'])

car_price1['enginelocation'] = Encoding.fit_transform(car_price1['enginelocation'])



car_price1['drivewheel'] = Encoding.fit_transform(car_price1['drivewheel'])

car_price1['carbody'] = Encoding.fit_transform(car_price1['carbody'])





car_price1.info()
#Lets have a look over data, how it got transformed using label encoding.

car_price1.head()
Correlation_matrix = car_price1.corr()

Plot_Correlation = Correlation_matrix['price'].sort_values(ascending=False)

Plot_Correlation.plot.bar()

# plot_Correlation.values.reshape(-1,1)

print(Plot_Correlation)



# Let us talk over negative correlation mpg (Miles Per gallon) is high negative correlted. Usually Car with high prices

# ranges come wiht less mileage reason being high body weight & high accelaration which compromises mileage. 

# In this correlation diagram we can see city mpg and highway mpg is high negatively correlated.



# Apart from this from postive correlation engine size, curbweight, horsepower,carsize(width and length) do play a 

# role voer the price of car range.  



# I beleive for Chinese car makers the below data should be enough for them to understand, think and decide

# car price. However lets think through machine learning models as well to predict the car prices and let us

# make chinese manufactures how these factors take a role on price prediction as well. 
# Let us try a simple regression over here. 

from sklearn.model_selection import train_test_split

X = car_price1.drop(columns=['price'])

y= car_price1['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42) 



from sklearn.linear_model import LinearRegression

Lreg = LinearRegression()

Lreg.fit(X_train,y_train)

predicted_price = Lreg.predict(X_test)



from sklearn.metrics import r2_score,mean_squared_error

print('SLR r2_score',r2_score(y_test,predicted_price))

print('SLR MSE',mean_squared_error(y_test,predicted_price))

print('Coefficients',Lreg.intercept_)



# High covariance can be seen. 

# Let us do try all other regression models and see which gives the highest covariance and compare the R2 score
# Choosing Random Forest Regressor as is one of the best models for regression predictions.



from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor(n_estimators=100)

RFR.fit(X_train,y_train)

RFR_Predict = RFR.predict(X_test)

print('Random forest r2_score',r2_score(y_test,RFR_Predict))

print('Random Forest MSE',mean_squared_error(y_test,RFR_Predict))



# could notice higher R2 score when compared to Simple Linear REgression 
#Lets try with XGB regressor model as well 

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

XGBR = XGBRegressor()

parameters = {'n_estimators': [200]}

xgb_grid = GridSearchCV(XGBR,parameters,cv = 2)

xgb_grid.fit(X_train,y_train)

print(xgb_grid.best_score_)

print(xgb_grid.best_params_)

XGBR_Predict = xgb_grid.predict(X_test)

print('XGB Regressor r2_score',r2_score(y_test,XGBR_Predict))

print('XGB Regressor MSE',mean_squared_error(y_test,XGBR_Predict))



# R2_Score seems to be less but XGB_grid score is fair enough.  
#Lets work through residuals for the best R2score. So in this case we can take up Random forest regressor 

# model 

Res = pd.DataFrame({'Actual': y_test, 'Predicted': RFR_Predict})

Res['Residual'] = Res['Actual'] - Res['Predicted']
Res.head()
 # Lets see residuals are distributed randomly or does it follow a pattern. \n",

 # let us try to plot a graph between predicted values and residuals\n"

    

x = Res['Predicted']

y = Res['Residual']

plt.scatter(x,y,color='blue')

plt.xlabel("Predicted")

plt.ylabel("Residuals")

plt.figure(figsize=(100,10))

plt.grid(True)

#observations are randomly distributed over zero line. few outliers could be seen, however as 

# as the majority of the data falls over in and around zeroHence the model looks to be good.  