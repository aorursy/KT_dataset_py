# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
%matplotlib inline
# Importing the traning set and test set
dataset = pd.read_csv(r'../input/train.csv')

y = dataset.iloc[:, -1].values
dataset_res = pd.read_csv(r'../input/test.csv')
y_res = dataset_res.iloc[:, -1].values
#basic checking the training set and test set
dataset.describe()

dataset_res.describe()
#By viewing the row count above, we know that some columes contain null value
#Checking the number of null values by column in training set 
zero_check_train = dataset.isnull().sum()
zero_check_train = zero_check_train[zero_check_train!=0]
zero_check_train = zero_check_train.sort_values(ascending=False)
zero_check_train
#show the correlation of features

corrmat = dataset.corr()
f, ax1 = plt.subplots(figsize=(12,9))

ax1=sns.heatmap(corrmat,vmax = 0.8);

 
#Show the top 10 feature which have strongest correlation with SalePrice
corr_sale = dataset.corr().SalePrice
corr_field = corr_sale.sort_values(ascending = False).head(11)
corr_field
#I will choose the these columns to predict the price using linear regression
#As checked the definition of those columns, there are some columns which are highly corrleated each other.Such as GarageCars and GarageArea, YearBuilt and YearRemodAdd We can consider dropping the columns further

corr_field = corr_field.drop(['YearRemodAdd','GarageCars','1stFlrSF']).index

#show the correlation of top 10 features 

corrmat = dataset[corr_field].corr()
f, ax1 = plt.subplots(figsize=(12,9))

ax1=sns.heatmap(corrmat,vmax = 0.8,annot = True);

corr_field = corr_field.drop('SalePrice');
#histogram
sns.distplot(y,fit = norm);
#skewness and kurtosis
print("Skewness: %f" % dataset['SalePrice'].skew())
print("Kurtosis: %f" % dataset['SalePrice'].kurt())
y_log = np.log(y)
sns.distplot(y_log,fit = norm);
for i in corr_field:
    plt.scatter(dataset[i],y_log)
    plt.xlabel(i)
    plt.ylabel("Salesprice")
    plt.show()
#Get the selected feature data and fill in the missing data with feature mean
X = dataset[corr_field]
X = X.fillna(X.mean())
X_res = dataset_res[corr_field]
X_res = X_res.fillna(X_res.mean())
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)
#using ramdom forest methord 

from sklearn.ensemble import RandomForestRegressor
reg_ran = RandomForestRegressor(n_estimators = 10, random_state = 0)
#Using GridSearchCV to find the optimimal parameter
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [10, 100, 1000], 'max_depth': [10,100,1000]}]
grid_search = GridSearchCV(estimator = reg_ran,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_log_error',
                           cv = 3,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train_log)

#Show the best score that model get
best_accuracy = grid_search.best_score_
best_accuracy
#Show the optimized value of parameters
best_parameters = grid_search.best_params_
best_parameters
# Evaluate the model with the test set
from sklearn.model_selection import cross_val_score
cross_val_score(reg_ran, X_test, y_test_log, scoring='neg_mean_squared_log_error') 

#Predict the result using the trained model
y_res_ran = np.exp(grid_search.predict(X_res))
cross_val_score(reg_ran, X_res, y_res_ran, scoring='neg_mean_squared_log_error') 

