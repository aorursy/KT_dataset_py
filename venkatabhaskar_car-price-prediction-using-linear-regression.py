# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
# Import the required packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# read the file 

car = pd.read_csv ("../input/CarPrice.csv")

new = car["CarName"].str.split(" ", n=1,expand = True)  

car["Carname"]= new[0]
car.drop(['CarName'], axis=1).head()
car.shape
car.info()
car.describe()
import matplotlib.pyplot as plt

import seaborn as sns
sns.pairplot(car)

plt.show()
# Dropping car Name after taking Car Company name and ignore the car model

car.drop(['CarName'], axis = 1, inplace = True)
varlist =  ['fueltype']



# Defining the map function

def binary_map(x):

    return x.map({"gas": 1, "diesel": 0})



# Applying the function to the car list

car[varlist] = car[varlist].apply(binary_map)
varlist =  ['aspiration']



# Defining the map function

def binary_map(x):

    return x.map({"std": 1, "turbo": 0})



# Applying the function to the car list

car[varlist] = car[varlist].apply(binary_map)
varlist =  ['doornumber']



# Defining the map function

def binary_map(x):

    return x.map({"two": 1, "four": 0})



# Applying the function to the car list

car[varlist] = car[varlist].apply(binary_map)
varlist =  ['enginelocation']



# Defining the map function

def binary_map(x):

    return x.map({"front": 1, "rear": 0})



# Applying the function to the car list

car[varlist] = car[varlist].apply(binary_map)
# Get the dummy variables for the feature 'carbody' and store it in a new variable - 'car_dm'

car_dm = pd.get_dummies(car['carbody'])
# Let's drop the first column from status df using 'drop_first = True'



car_dm = pd.get_dummies(car['carbody'], drop_first = True)
# Add the results to the original car dataframe



car = pd.concat([car, car_dm], axis = 1)
# Drop 'carbody' as we have created the dummies for it



car.drop(['carbody'], axis = 1, inplace = True)
# Get the dummy variables for the feature 'drivewheel' and store it in a new variable - 'car_dm'

car_dm = pd.get_dummies(car['drivewheel'])
car_dm.head()

# Let's drop the first column from status df using 'drop_first = True'



car_dm1 = pd.get_dummies(car['drivewheel'], drop_first = True)
# Add the results to the original car dataframe



car = pd.concat([car, car_dm1], axis = 1)
# Drop 'drivewheel' as we have created the dummies for it



car.drop(['drivewheel'], axis = 1, inplace = True)
# Get the dummy variables for the feature 'enginetype' and store it in a new variable - 'car_dm'

car_dm = pd.get_dummies(car['enginetype'])
car_dm.head()
# Let's drop the first column from status df using 'drop_first = True'



car_dm = pd.get_dummies(car['enginetype'], drop_first = True)
# Add the results to the original housing dataframe



car = pd.concat([car, car_dm], axis = 1)
# Drop 'enginetype' as we have created the dummies for it



car.drop(['enginetype'], axis = 1, inplace = True)


# Get the dummy variables for the feature 'cylindernumber' and store it in a new variable - 'car_dm'

car_dm = pd.get_dummies(car['cylindernumber'])
# Let's drop the first column from status df using 'drop_first = True'



car_dm = pd.get_dummies(car['cylindernumber'], drop_first = True)
# Add the results to the original car dataframe



car = pd.concat([car, car_dm], axis = 1)
# Drop 'cylindernumber' as we have created the dummies for it



car.drop(['cylindernumber'], axis = 1, inplace = True)
# Get the dummy variables for the feature 'fuelsystem' and store it in a new variable - 'car_dm'

car_dm = pd.get_dummies(car['fuelsystem'])

# Let's drop the first column from status df using 'drop_first = True'



car_dm = pd.get_dummies(car['fuelsystem'], drop_first = True)


# Add the results to the original car dataframe



car = pd.concat([car, car_dm], axis = 1)
# Drop 'fuelsystem' as we have created the dummies for it



car.drop(['fuelsystem'], axis = 1, inplace = True)

car.drop(['car_ID'], axis = 1, inplace = True)
# Get the dummy variables for the feature 'Carname' and store it in a new variable - 'car_dm'

car_dm = pd.get_dummies(car['Carname'])

# Let's drop the first column from status df using 'drop_first = True'



car_dm = pd.get_dummies(car['Carname'], drop_first = True)


# Add the results to the original car dataframe



car = pd.concat([car, car_dm], axis = 1)
# Drop 'carbody' as we have created the dummies for it



car.drop(['Carname'], axis = 1, inplace = True)

car.head()
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

df_train, df_test = train_test_split(car, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight','curbweight','enginesize','boreratio','stroke','compressionratio'

           ,'horsepower','peakrpm','citympg','highwaympg','price']



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
y_train = df_train.pop('price')

X_train = df_train
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 70

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 70)             # running RFE

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# RFE NOT SATISFIED VARIables



col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
# Creating X_train_rfe dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
#Let's see the summary of our linear model

print(lm.summary())
# drop NAN and typo variables 



X_train_new = X_train_rfe.drop(["volkswagen","vokswagen","toyouta","toyota"], axis = 1)

# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Calculate the VIFs again for the new model

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Drop VIF > 5 impacted variables



X_train_new = X_train_new.drop(["rotor","l","fueltype","peugeot","idi","enginelocation","two","three","subaru","ohcf","four"

                               ,"enginesize","compressionratio","six","five","boreratio","horsepower","mpfi","citympg","curbweight"

                               ,"highwaympg","2bbl","sedan","hatchback","ohc","carlength","wheelbase","carwidth","wagon","rwd"], axis = 1)
# Adding a constant variable and re-run the model

import statsmodels.api as sm  

X_train_lm_n = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_lm_n).fit()   # Running the linear model
#Let's see the summary of our linear model

print(lm.summary())
X_train_lm_n.columns
# not variable which is not significant



X_train_new1 = X_train_lm_n.drop(["symboling"], axis = 1)
# Calculate the VIFs again for the new model

vif = pd.DataFrame()

vif['Features'] = X_train_new1.columns

vif['VIF'] = [variance_inflation_factor(X_train_new1.values, i) for i in range(X_train_new1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Drop the variables which has VIF > 5 and re-run the model

X_train_new2 = X_train_new1.drop(["mfi","spfi","mercury","porcshce"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm_n2 = sm.add_constant(X_train_new2)
lm = sm.OLS(y_train,X_train_lm_n2).fit()   # Running the linear model
print(lm.summary())
# Drop the variables which are not significant P values



X_train_new3 = X_train_lm_n2.drop(['spdi','vw','hardtop','maxda','mazda','twelve'], axis = 1)

# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm_n3 = sm.add_constant(X_train_new3)
lm = sm.OLS(y_train,X_train_lm_n3).fit()   # Running the linear model
print(lm.summary())
# Drop variables which are not signicant > P > 0.05 

X_train_new4 = X_train_lm_n3.drop(['isuzu','renault','dohcv','4bbl','alfa-romero','chevrolet','nissan','peakrpm','honda',

                                   'doornumber','carheight','mitsubishi','plymouth','dodge'], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm_n4 = sm.add_constant(X_train_new4)
lm = sm.OLS(y_train,X_train_lm_n4).fit()   # Running the linear model
print(lm.summary())
X_train_lm_n4.columns
# Now predict the model on train dataset and do error residuals plot to test for idenpendence and normality 

y_train_price = lm.predict(X_train_lm_n4)
# Importing the required libraries for plots.

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight','curbweight','enginesize','boreratio','stroke','compressionratio'

           ,'horsepower','peakrpm','citympg','highwaympg','price']



df_test[num_vars] = scaler.transform(df_test[num_vars])
y_test = df_test.pop('price')

X_test = df_test
X_train_lm_n4.columns
# Now let's use our model to make predictions.

X_train_lm_n5 = X_train_lm_n4.drop(['const'], axis = 1)



# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_lm_n5.columns]



X_test_new.columns

# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions on test dataset

y_pred = lm.predict(X_test_new)
print(lm.summary())
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)                          # Y-label
X_test_new.columns
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)