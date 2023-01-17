# to handle datasets

import pandas as pd

import numpy as np



# for plotting

import matplotlib.pyplot as plt

%matplotlib inline



# to divide train and test set

from sklearn.model_selection import train_test_split



# feature scaling

from sklearn.preprocessing import MinMaxScaler



# for tree binarisation

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score





# to build the models

from sklearn.linear_model import LinearRegression, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

import xgboost as xgb



# to evaluate the models

from sklearn.metrics import mean_squared_error

from math import sqrt



pd.pandas.set_option('display.max_columns', None)



import warnings

warnings.filterwarnings('ignore')
# load dataset

data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

print(data.shape)

data.head()
# Load the dataset for submission (the one on which our model will be evaluated by Kaggle)

# it contains exactly the same variables, but not the target



submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

submission.head()
# let's inspect the type of variables in pandas

data.dtypes
# we also have an Id variable, that we shoulld not use for predictions:



print('Number of House Id labels: ', len(data.Id.unique()))

print('Number of Houses in the Dataset: ', len(data))
# find categorical variables

categorical = [var for var in data.columns if data[var].dtype=='O']

print('There are {} categorical variables'.format(len(categorical)))
# make a list of the numerical variables first

numerical = [var for var in data.columns if data[var].dtype!='O']



# list of variables that contain year information

year_vars = [var for var in numerical if 'Yr' in var or 'Year' in var]



year_vars
data[year_vars].head()
data.groupby('MoSold')['SalePrice'].median().plot()

plt.title('House price variation in the year')

plt.ylabel('mean House price')
# let's visualise the values of the discrete variables

discrete = []



for var in numerical:

    if len(data[var].unique())<20 and var not in year_vars:

        print(var, ' values: ', data[var].unique())

        discrete.append(var)

print()

print('There are {} discrete variables'.format(len(discrete)))
# find continuous variables

# let's remember to skip the Id variable and the target variable SalePrice, which are both also numerical



numerical = [var for var in numerical if var not in discrete and var not in ['Id', 'SalePrice'] and var not in year_vars]

print('There are {} numerical and continuous variables'.format(len(numerical)))
# let's visualise the percentage of missing values for each variable

for var in data.columns:

    if data[var].isnull().sum()>0:

        print(var, data[var].isnull().mean())
# let's now determine how many variables we have with missing information



vars_with_na = [var for var in data.columns if data[var].isnull().sum()>0]

print('Total variables that contain missing information: ', len(vars_with_na))
# let's inspect the type of those variables with a lot of missing information

for var in data.columns:

    if data[var].isnull().mean()>0.80:

        print(var, data[var].unique())
# let's look at the numerical variables

numerical
# let's make boxplots to visualise outliers in the continuous variables 

# and histograms to get an idea of the distribution



for var in numerical:

    plt.figure(figsize=(15,6))

    plt.subplot(1, 2, 1)

    fig = data.boxplot(column=var)

    fig.set_title('')

    fig.set_ylabel(var)

    

    plt.subplot(1, 2, 2)

    fig = data[var].hist(bins=20)

    fig.set_ylabel('Number of houses')

    fig.set_xlabel(var)



    plt.show()
# outlies in discrete variables

for var in discrete:

    (data.groupby(var)[var].count() / np.float(len(data))).plot.bar()

    plt.ylabel('Percentage of observations per label')

    plt.title(var)

    plt.show()

    #print(data[var].value_counts() / np.float(len(data)))

    print()
no_labels_ls = []

for var in categorical:

    no_labels_ls.append(len(data[var].unique()))

    

 

tmp = pd.Series(no_labels_ls)

tmp.index = pd.Series(categorical)

tmp.plot.bar(figsize=(12,8))

plt.title('Number of categories in categorical variables')

plt.xlabel('Categorical variables')

plt.ylabel('Number of different categories')
# Let's separate into train and test set



X_train, X_test, y_train, y_test = train_test_split(data, data.SalePrice, test_size=0.1,

                                                    random_state=0)

X_train.shape, X_test.shape
# function to calculate elapsed time



def elapsed_years(df, var):

    # capture difference between year variable and year the house was sold

    df[var] = df['YrSold'] - df[var]

    return df
for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:

    X_train = elapsed_years(X_train, var)

    X_test = elapsed_years(X_test, var)

    submission = elapsed_years(submission, var)
X_train[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
# drop YrSold

X_train.drop('YrSold', axis=1, inplace=True)

X_test.drop('YrSold', axis=1, inplace=True)

submission.drop('YrSold', axis=1, inplace=True)
# print variables with missing data

# keep in mind that now that we created those new temporal variables, we

# are going to treat them as numerical and continuous as well:



# remove YrSold because it is no longer in our dataset

year_vars.remove('YrSold')



# examine percentage of missing values

for col in numerical+year_vars:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
# add variable indicating missingness + median imputation

for df in [X_train, X_test, submission]:

    for var in ['LotFrontage', 'GarageYrBlt']:

        df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)

        df[var].fillna(X_train[var].median(), inplace=True) 



for df in [X_train, X_test, submission]:

    df['MasVnrArea'].fillna(X_train.MasVnrArea.median(), inplace=True)
# print variables with missing data

for col in discrete:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
# print variables with missing data

for col in categorical:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
# add label indicating 'Missing' to categorical variables



for df in [X_train, X_test, submission]:

    for var in categorical:

        df[var].fillna('Missing', inplace=True)
# check absence of null values

for var in X_train.columns:

    if X_train[var].isnull().sum()>0:

        print(var, X_train[var].isnull().sum())
# check absence of null values

for var in X_train.columns:

    if X_test[var].isnull().sum()>0:

        print(var, X_test[var].isnull().sum())
# check absence of null values

submission_vars = []

for var in X_train.columns:

    if var!='SalePrice' and submission[var].isnull().sum()>0:

        print(var, submission[var].isnull().sum())

        submission_vars.append(var)
# Fill NA with median value for those variables that show NA only in the submission set



for var in submission_vars:

    submission[var].fillna(X_train[var].median(), inplace=True)
def tree_binariser(var):

    score_ls = [] # here I will store the mse



    for tree_depth in [1,2,3,4]:

        # call the model

        tree_model = DecisionTreeRegressor(max_depth=tree_depth)



        # train the model using 3 fold cross validation

        scores = cross_val_score(tree_model, X_train[var].to_frame(), y_train, cv=3, scoring='neg_mean_squared_error')

        score_ls.append(np.mean(scores))



    # find depth with smallest mse

    depth = [1,2,3,4][np.argmin(score_ls)]

    #print(score_ls, np.argmin(score_ls), depth)



    # transform the variable using the tree

    tree_model = DecisionTreeRegressor(max_depth=depth)

    tree_model.fit(X_train[var].to_frame(), X_train.SalePrice)

    X_train[var] = tree_model.predict(X_train[var].to_frame())

    X_test[var] = tree_model.predict(X_test[var].to_frame())

    submission[var] =  tree_model.predict(submission[var].to_frame())
for var in numerical:

    tree_binariser(var)
X_train[numerical].head()
# let's explore how many different buckets we have now among our engineered continuous variables

for var in numerical:

    print(var, len(X_train[var].unique()))
for var in numerical:

    X_train.groupby(var)['SalePrice'].mean().plot.bar()

    plt.title(var)

    plt.ylabel('Mean House Price')

    plt.xlabel('Discretised continuous variable')

    plt.show()
def rare_imputation(variable):

    # find frequent labels / discrete numbers

    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))

    frequent_cat = [x for x in temp.loc[temp>0.03].index.values]

    

    X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], 'Rare')

    X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], 'Rare')

    submission[variable] = np.where(submission[variable].isin(frequent_cat), submission[variable], 'Rare')
# the following vars in the submission dataset are encoded in different types

# so first I cast them as int, like in the train set



for var in ['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']:

    submission[var] = submission[var].astype('int')
# find infrequent labels in categorical variables and replace by Rare

for var in categorical:

    rare_imputation(var)

    

# find infrequent labels in categorical variables and replace by Rare

# remember that we are treating discrete variables as if they were categorical

for var in discrete:

    rare_imputation(var)
# check that we haven't created missing values in the submission dataset

for var in X_train.columns:

    if var!='SalePrice' and submission[var].isnull().sum()>0:

        print(var, submission[var].isnull().sum())

        submission_vars.append(var)
# let's check that it worked

for var in categorical:

    (X_train.groupby(var)[var].count() / np.float(len(X_train))).plot.bar()

    plt.ylabel('Percentage of observations per label')

    plt.title(var)

    plt.show()
# let's check that it worked

for var in discrete:

    (X_train.groupby(var)[var].count() / np.float(len(X_train))).plot.bar()

    plt.ylabel('Percentage of observations per label')

    plt.title(var)

    plt.show()
def encode_categorical_variables(var, target):

        # make label to price dictionary

        ordered_labels = X_train.groupby([var])[target].mean().to_dict()

        

        # encode variables

        X_train[var] = X_train[var].map(ordered_labels)

        X_test[var] = X_test[var].map(ordered_labels)

        submission[var] = submission[var].map(ordered_labels)



# encode labels in categorical vars

for var in categorical:

    encode_categorical_variables(var, 'SalePrice')

    

# encode labels in discrete vars

for var in discrete:

    encode_categorical_variables(var, 'SalePrice')
# sanity check: let's see that we did not introduce NA by accident

for var in X_train.columns:

    if var!='SalePrice' and submission[var].isnull().sum()>0:

        print(var, submission[var].isnull().sum())
#let's inspect the dataset

X_train.head()
X_train.describe()
# let's create a list of the training variables

training_vars = [var for var in X_train.columns if var not in ['Id', 'SalePrice']]



print('total number of variables to use for training: ', len(training_vars))
training_vars
# fit scaler

scaler = MinMaxScaler() # create an instance

scaler.fit(X_train[training_vars]) #  fit  the scaler to the train set for later use
xgb_model = xgb.XGBRegressor()



eval_set = [(X_test[training_vars], np.log(y_test))]

xgb_model.fit(X_train[training_vars], np.log(y_train), eval_set=eval_set, verbose=False)



pred = xgb_model.predict(X_train[training_vars])

print('xgb train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))

print('xgb train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))

print()

pred = xgb_model.predict(X_test[training_vars])

print('xgb test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))

print('xgb test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))
rf_model = RandomForestRegressor(n_estimators=800, max_depth=6)

rf_model.fit(X_train[training_vars], np.log(y_train))



pred = rf_model.predict(X_train[training_vars])

print('rf train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))

print('rf train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))



print()

pred = rf_model.predict(X_test[training_vars])

print('rf test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))

print('rf test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))
SVR_model = SVR()

SVR_model.fit(scaler.transform(X_train[training_vars]), np.log(y_train))



pred = SVR_model.predict(X_train[training_vars])

print('SVR train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))

print('SVR train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))



print()

pred = SVR_model.predict(X_test[training_vars])

print('SVR test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))

print('SVR test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))
lin_model = Lasso(random_state=2909, alpha=0.005)

lin_model.fit(scaler.transform(X_train[training_vars]), np.log(y_train))



pred = lin_model.predict(scaler.transform(X_train[training_vars]))

print('Lasso Linear Model train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))

print('Lasso Linear Model train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))



print()

pred = lin_model.predict(scaler.transform(X_test[training_vars]))

print('Lasso Linear Model test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))

print('Lasso Linear Model test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))
# make predictions for the submission dataset

final_pred = pred = lin_model.predict(scaler.transform(submission[training_vars]))
temp = pd.concat([submission.Id, pd.Series(np.exp(final_pred))], axis=1)

temp.columns = ['Id', 'SalePrice']

temp.head()
temp.to_csv('submit_housesale.csv', index=False)