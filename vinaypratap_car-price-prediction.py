# Importing all required packages
import numpy as np
import pandas as pd

# for plotting purpose
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# For splitting the data into train and test
from sklearn.model_selection import train_test_split

# For reskaling purpose
from sklearn.preprocessing import MinMaxScaler

# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')
# Reading Car price file
carprice = pd.read_csv('../input/CarPrice_Assignment.csv')

# Lets look into car price data 
print(carprice.shape)
carprice.info()
carprice.head(2)

# Total 205 rows & 26 columns present in dataset. 
# Fortunately there are No null values.
# Lets check basic statistics of dataset

carprice.describe()
# Lets prepare data for further analysis
# Need to pick only name of Car from CarName column of the dataset
# Lets look first unique values of CarName

carprice.CarName.unique()

# We could notice that '-' present with few car names.
# lets get rid of '-' and change it to space so that we could pick first word from CarName column and use space as dilimiter.
# There are few car names which are written incorrectly & VW in short form of volkswagen.


# Get rid of '-' from CarName column. We will same column name for further analysis.

carprice["CarName"] = carprice["CarName"].str.replace('-', ' ')

# correct few incorrect names of Cars
carprice["CarName"] = carprice["CarName"].str.replace('maxda', 'mazda')
carprice["CarName"] = carprice["CarName"].str.replace('Nissan', 'nissan')
carprice["CarName"] = carprice["CarName"].str.replace('porcshce', 'porsche')
carprice["CarName"] = carprice["CarName"].str.replace('toyouta', 'toyota')
carprice["CarName"] = carprice["CarName"].str.replace('vw', 'volkswagen')
carprice["CarName"] = carprice["CarName"].str.replace('vokswagen', 'volkswagen')

# Lets pick only first word from carName column
carprice["CarCompanyName"] = carprice["CarName"].str.split().str[0]

# Lets look for unique columns
carprice.CarCompanyName.unique()
# Looks good now.
# Find Duplicates rows if any

carprice.loc[carprice.duplicated()]

# There are no duplicate rows as no rows were printed in output.
# plot a pair plot

#sns.pairplot(carprice)
#plt.show()

# Carname vs price

plt.figure(figsize=(20, 8))
sns.boxplot(x = 'CarCompanyName', y = 'price', data = carprice)
plt.xticks(rotation = 90)
plt.show()

# Jaguar,Buick,Porsche have highest average price in all cars.
# Lets look some more variables aganist Price

plt.figure(figsize=(20, 12))
plt.subplot(2,2,1)
sns.boxplot(x = 'symboling', y = 'price', data = carprice)
plt.subplot(2,2,2)
sns.boxplot(x = 'fueltype', y = 'price', data = carprice)
plt.subplot(2,2,3)
sns.boxplot(x = 'enginetype', y = 'price', data = carprice)
plt.xticks(rotation = 90)
plt.subplot(2,2,4)
sns.boxplot(x = 'carbody', y = 'price', data = carprice)
plt.xticks(rotation = 90)
plt.show()
# Lets look for rest of the categorical variables

plt.figure(figsize=(28, 13))
plt.subplot(2,3,1)
sns.boxplot(x = 'enginelocation', y = 'price', data = carprice)
plt.subplot(2,3,2)
sns.boxplot(x = 'fuelsystem', y = 'price', data = carprice)
plt.subplot(2,3,3)
sns.boxplot(x = 'cylindernumber', y = 'price', data = carprice)
plt.subplot(2,3,4)
sns.boxplot(x = 'aspiration', y = 'price', data = carprice)
plt.subplot(2,3,5)
sns.boxplot(x = 'doornumber', y = 'price', data = carprice)
plt.subplot(2,3,6)
sns.boxplot(x = 'drivewheel', y = 'price', data = carprice)
plt.show()
# Lets drop few columns which have no significance in our analysis. column Car_ID
carprice.drop(['car_ID'], inplace = True, axis =1)

# Lets drop CarName as well. We ahve already picked car company name from it & now this column has no use.
carprice.drop(['CarName'], inplace = True, axis =1)

# Applying the map function to the fueltype
carprice['fueltype'] = carprice['fueltype'].map({'gas': 1, 'diesel': 0})
# Applying the map function to the aspiration
carprice['aspiration'] = carprice['aspiration'].map({'std': 1, 'turbo': 0})
# Applying the map function to the doornumber
carprice['doornumber'] = carprice['doornumber'].map({'two': 1, 'four': 0})
# Applying the map function to the enginelocation
carprice['enginelocation'] = carprice['enginelocation'].map({"front": 1, "rear": 0})
carprice.head()
# Get the dummy variables for the feature 'carbody' and store it in a new variable - 'body_dummy'
body_dummy = pd.get_dummies(carprice['carbody'])
body_dummy.head(5) 
# Let's drop the first column from status df using 'drop_first = True'
body_dummy = pd.get_dummies(carprice['carbody'],drop_first=True)

# Add the results to the original carprice dataframe
carprice = pd.concat([carprice, body_dummy], axis = 1)
carprice.head()
# Drop 'carbody' as we have created the dummies for it
carprice.drop(['carbody'], axis = 1, inplace = True)
carprice.head()
# Get the dummy variables for the feature 'carbody' and store it in a new variable - 'wheel_dummy'
wheel_dummy = pd.get_dummies(carprice['drivewheel'])
wheel_dummy.head(5) 
# Let's drop the first column from status df using 'drop_first = True'
wheel_dummy = pd.get_dummies(carprice['drivewheel'], drop_first = True)
# Add the results to the original carprice dataframe
carprice = pd.concat([carprice, wheel_dummy], axis = 1)
carprice.head()
# Drop 'carbody' as we have created the dummies for it
carprice.drop(['drivewheel'], axis = 1, inplace = True)
carprice.head()
# Get the dummy variables for the feature 'enginetype' and store it in a new variable - 'enginetype_dummy'
enginetype_dummy = pd.get_dummies(carprice['enginetype'])
enginetype_dummy.head()
# Let's drop the first column from enginetype_dummy df using 'drop_first = True'
enginetype_dummy = pd.get_dummies(carprice['enginetype'],drop_first = True)
# Add the results to the original carprice dataframe
carprice = pd.concat([carprice, enginetype_dummy], axis = 1)
carprice.head()
# Drop 'enginetype' as we have created the dummies for it
carprice.drop(['enginetype'], axis = 1, inplace = True)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
carprice['cylindernumber'] = le.fit_transform(carprice.cylindernumber.values)
# Get the dummy variables for the feature 'enginetype' and store it in a new variable - 'enginetype_dummy'
#cylindernumber_dummy = pd.get_dummies(carprice['cylindernumber'])
#cylindernumber_dummy.head()
# Let's drop the first column from status df using 'drop_first = True'
#cylindernumber_dummy = pd.get_dummies(carprice['cylindernumber'], drop_first = True)
# Add the results to the original carprice dataframe
#carprice = pd.concat([carprice, cylindernumber_dummy], axis = 1)
carprice['cylindernumber'].head(5)
# Drop 'cylindernumber' as we have created the dummies for it
# carprice.drop(['cylindernumber'], axis = 1, inplace = True)

# Get the dummy variables for the feature 'fuelsystem' and store it in a new variable - 'fuelsystem_dummy'
fuelsystem_dummy = pd.get_dummies(carprice['fuelsystem'])
fuelsystem_dummy.head()
# Let's drop the first column from fuelsystem_dummy df using 'drop_first = True'
fuelsystem_dummy = pd.get_dummies(carprice['fuelsystem'], drop_first = True)
# Add the results to the original carprice dataframe
carprice = pd.concat([carprice, fuelsystem_dummy], axis = 1)
carprice.head()
# Drop 'fuelsystem' as we have created the dummies for it
carprice.drop(['fuelsystem'], axis = 1, inplace = True)

# Get the dummy variables for the feature 'CarName' and store it in a new variable - 'CarName_dummy'
CarName_dummy = pd.get_dummies(carprice['CarCompanyName'])
CarName_dummy.head()
# Let's drop the first column from CarName_dummy df using 'drop_first = True'
CarName_dummy = pd.get_dummies(carprice['CarCompanyName'], drop_first = True)
# Add the results to the original carprice dataframe
carprice = pd.concat([carprice, CarName_dummy], axis = 1)
carprice.head()
# Drop 'fuelsystem' as we have created the dummies for it
carprice.drop(['CarCompanyName'], axis = 1, inplace = True)
print(carprice.shape)
carprice.head()
# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(carprice, train_size = 0.7, test_size = 0.3, random_state = 100)
scaler = MinMaxScaler()
# Apply scaler() to all the columns except the binary and 'dummy' variables
num_vars = ['symboling', 'carlength', 'carheight', 'enginesize', 'boreratio','stroke', 'compressionratio', 'horsepower',
'peakrpm', 'citympg', 'price','carwidth','curbweight','highwaympg','wheelbase','cylindernumber']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
df_train.describe()
y_train = df_train.pop('price')
X_train = df_train
# Running RFE with the output number of the variable equal to 20

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 15)                          # running RFE
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]
col
X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables
X_train_1 = X_train[col]
# Adding a constant variable 
import statsmodels.api as sm  
X_train_1 = sm.add_constant(X_train_1)
lm = sm.OLS(y_train,X_train_1).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# Lets drop Buick which has high p value .142
X_train_2 = X_train_1.drop(["peakrpm"], axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_2 = sm.add_constant(X_train_2)
lm = sm.OLS(y_train,X_train_2).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model after dropping carname buick
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_2
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
#Correlation using heatmap
plt.figure(figsize = (20, 10))
sns.heatmap(X_train_2.corr(), annot = True, cmap="YlGnBu")
plt.show()
X_train_3 = X_train_2.drop(["enginelocation"], axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_3 = sm.add_constant(X_train_3)
lm = sm.OLS(y_train,X_train_3).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_3
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# boreration is highly correlated with enginesize & carlength. Lets drop it
X_train_4 = X_train_3.drop(["carheight"], axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_4 = sm.add_constant(X_train_4)
lm = sm.OLS(y_train,X_train_4).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_4
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_5 = X_train_4.drop(["mitsubishi"], axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_5 = sm.add_constant(X_train_5)
lm = sm.OLS(y_train,X_train_5).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_5
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_6 = X_train_5.drop(["renault"], axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_6 = sm.add_constant(X_train_6)
lm = sm.OLS(y_train,X_train_6).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_6
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_7 = X_train_6.drop(["cylindernumber"], axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_7 = sm.add_constant(X_train_7)
lm = sm.OLS(y_train,X_train_7).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_7
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
#Correlation using heatmap
plt.figure(figsize = (11, 4))
sns.heatmap(X_train_7.corr(), annot = True, cmap="YlGnBu")
plt.show()
X_train_8 = X_train_7.drop(["curbweight"], axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_8 = sm.add_constant(X_train_8)
lm = sm.OLS(y_train,X_train_8).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_8
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_9 = X_train_8.drop(["l"], axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_9 = sm.add_constant(X_train_9)
lm = sm.OLS(y_train,X_train_9).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_9
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_10 = X_train_9.drop(["peugeot"], axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_10 = sm.add_constant(X_train_10)
lm = sm.OLS(y_train,X_train_10).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_10
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_11 = X_train_10.drop(["ohcf"], axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_11 = sm.add_constant(X_train_11)
lm = sm.OLS(y_train,X_train_11).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_11
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_12 = X_train_11.drop(["subaru"], axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_12 = sm.add_constant(X_train_12)
lm = sm.OLS(y_train,X_train_12).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_12
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
y_train_price = lm.predict(X_train_12)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 5)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label
# Apply scaler() to all the columns except the binary and 'dummy' variables
num_vars = ['symboling', 'carlength', 'carheight', 'enginesize', 'boreratio','stroke', 'compressionratio', 'horsepower',
'peakrpm', 'citympg', 'price','carwidth','curbweight','highwaympg','wheelbase','cylindernumber']
df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.describe()
y_test = df_test.pop('price')
X_test = df_test
X_train_new = X_train_12.drop(["const"], axis = 1)

vif = pd.DataFrame()
X = X_train_12
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Creating X_test_new dataframe by dropping variables from X_test & picking only columns which were in train model.
X_test_new = X_test[X_train_new.columns]
X_test_new.info()
# Adding constant variable to test dataframe
X_test_new = sm.add_constant(X_test_new)
# Making predictions
y_pred = lm.predict(X_test_new)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label
# Actual vs Predicted
c = [i for i in range(1,63,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=3.5, linestyle="-")     #Plotting Actual
plt.plot(c,y_pred, color="red",  linewidth=3.5, linestyle="-")  #Plotting predicted
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Car Price', fontsize=16)  
# Error terms
fig = plt.figure()
c = [i for i in range(1,63,1)]
# plt.plot(c,y_test-y_pred_m9, color="blue", linewidth=2.5, linestyle="-")
plt.scatter(c,y_test-y_pred)

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)  
# Plotting the error terms to understand the distribution.
fig = plt.figure()
sns.distplot((y_test-y_pred),bins=5)
fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 
plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label
plt.ylabel('Index', fontsize=16)  