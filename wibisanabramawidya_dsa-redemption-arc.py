# Import libraries

from datetime import datetime

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression, Ridge

from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from catboost import CatBoostRegressor



import warnings

warnings.filterwarnings("ignore")



# Set infinite max rows since data has 89 Columns

# Showing all rows makes interpreting data easier

# pd.set_option('display.max_rows', None)

# pd.set_option('display.max_colwidth', None)
# Check input files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load data

df = pd.read_csv('/kaggle/input/fifa19/data.csv')

df.info()
# Player position raking columns

pos_col = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']



# Some rows contain no data on position rating

df[pos_col].isna().sum()
# Rows that have no position ratings are the same rows as each other

df[df[pos_col].isna().all(axis=1)][pos_col]
gk_pos_count  = df[df[pos_col[1:]].isna().all(axis=1)][df.Position == 'GK'].shape[0]

nan_pos_count = df[df[pos_col[1:]].isna().all(axis=1)][df.Position.isna()].shape[0]



print('GK  :', gk_pos_count)

print('NaN :', nan_pos_count)
# Value related columns

val_col = ['Club', 'Value', 'Wage']



# Some rows contains no data on club, value, and wage

# Value and wage will not be NaN, but be valued at 0

bad_club  = df['Club'].isna()

bad_value = df['Value'].str.match('€0')

bad_wage  = df['Wage'].str.match('€0')



pd.DataFrame(data={'Club': bad_club, 'Value': bad_value, 'Wage': bad_wage}).sum()
df[df['Club'].isna()][val_col]
# Rows with bad position rating data will be dropped

# Position rating is a row we use for some preprocessing

df = df.dropna(subset=pos_col)



# Rows with bad Club, Value, and Wage will be dropped

# Players without club is assumed to not play in this game

# Value and wage is 0, being an anomaly

df = df.dropna(subset=val_col)



df.shape
# Create copy of dataframe for cleaner processing

data = df.iloc[:]
# Split Work Rate column into AWR and DWR

data = data.drop(['Work Rate'], axis=1, errors='ignore')

data['AWR'] = df['Work Rate'].apply(lambda x: x.split('/')[0].strip())

data['DWR'] = df['Work Rate'].apply(lambda x: x.split('/')[1].strip())



data[['AWR', 'DWR']].head()
# Parse Date as unix

# Accepted format is `<abbr_month> <date>, <year>`

# example `Jan 1, 2000`

def date2unix(d):

    d = datetime.strptime(d, '%b %d, %Y')

    return int(d.timestamp())
# Parse joined value to unix timestamp

data = data.drop(['Joined'], axis=1, errors='ignore')

data['joined_val'] = df['Joined'].fillna("Jan 1, 1970").apply(date2unix)



data[['joined_val']].head()
# Height represented in typical feet'inch notation

# Convert to inches, since units does not matter

def height2int(h):

    feet, inch = h.split("'")

    return int(feet) * 12 + int(inch)



# Weight represented in lbs

# Truncate lbs suffix

def weight2int(w):

    return int(w[:-3])
# Parse height to numeric data

data = data.drop(['Height'], axis=1, errors='ignore')

data['height_val'] = df['Height'].fillna("0'0").apply(height2int)



# Parse weight to numeric data

data = data.drop(['Weight'], axis=1, errors='ignore')

data['weight_val'] = df['Weight'].fillna('0lbs').apply(weight2int)



data[['height_val', 'weight_val']].head()
# Ratings are represented with x+y

# x is the base rating

# y is the modifier when the player is in a good mood

# For the purpose of this kernel we can use the base rating

def rating2int(r):

    base, _ = r.split('+')

    return int(base)
# Transform position rating values to integer

for col in pos_col:

    data = data.drop([col], axis=1, errors='ignore')

    data[col] = df[col].fillna('0+0').apply(rating2int)



data[pos_col].head()
# Conversion function from currency to int

# Currency syntaxin this dataset is represented as such

# €1M, €10K, €0, etc

def money2int(m):

    # Cut out Euro sign prefix

    if m.startswith('€'):

        m = m[1:]

    

    # Get multiplier suffix for thousands and millions

    multi = 1

    if m.endswith('K'):

        multi = 1000

        m = m[:-1]

    if m.endswith('M'):

        multi = 1000000

        m = m[:-1]

    

    val = float(m) * multi

    return int(val)
# Transform currency columns to integer

data = data.drop(['Value', 'Wage', 'Release Clause'], axis=1, errors='ignore')

data['value_val']          = df['Value'].fillna('€0').apply(money2int)

data['wage_val']           = df['Wage'].fillna('€0').apply(money2int)

data['release_clause_val'] = df['Release Clause'].fillna('€0').apply(money2int)



data[['value_val', 'wage_val', 'release_clause_val']].head()
# Show correlation to justify dropping a significant column

data[['value_val', 'wage_val', 'release_clause_val']].corr()
# Delete Release Clause

data = data.drop(['release_clause_val'], axis=1, errors='ignore')
# Ignore image fields

data = data.drop(['Photo', 'Flag', 'Club Logo'], axis=1, errors='ignore')



# Ignore identitier fields with high cardinality

data = data.drop(['Unnamed: 0', 'ID', 'Name'], axis=1, errors='ignore')



# Ignore irrelevant data

data = data.drop(['Real Face', 'Jersey Number', 'Loaned From', 'Contract Valid Until'], axis=1, errors='ignore')
data.info()
# Create copy of dataset to preserve original

data_naive = data.iloc[:]
# Get one hot encoding for naive categorical attributes

naive_attr = ['Nationality', 'Club', 'Preferred Foot', 'Body Type', 'Position', 'AWR', 'DWR']

naive_dummies = pd.get_dummies(data[naive_attr])



naive_dummies.head()
# Append body type encoding to dataset

data_naive = data_naive.drop(naive_attr, axis=1, errors='ignore')

data_naive = data_naive.drop(naive_dummies.columns, axis=1, errors='ignore')

data_naive = data_naive.join(naive_dummies)



data_naive.head()
data_naive.info()
# Split data into attributes and targets

X  = data_naive.drop(['value_val', 'wage_val'], axis=1)

y1 = data_naive['value_val']

y2 = data_naive['wage_val']



X1_train, X1_test , y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=420)

X2_train, X2_test , y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=420)



X2_train.info()
# Perform linear regression on player Value

lin_reg1 = LinearRegression(normalize=True)



run_time = time.time()

lin_reg1.fit(X1_train, y1_train)

lin_pred1_train = lin_reg1.predict(X1_train)

lin_pred1_test = lin_reg1.predict(X1_test)

run_time = time.time() - run_time



print('NAIVE LINEAR REGRESSION')

print('PREDICTING PLAYER VALUE')

print()



lin_coef = pd.Series(lin_reg1.coef_, index=X1_train.columns)

print('Most significant Coef:')

print(lin_coef.reindex(lin_coef.abs().sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y1_train, lin_pred1_train))

print('R2 Score TEST   :', r2_score(y1_test, lin_pred1_test))

print('MSE Score TRAIN :', mean_squared_error(y1_train, lin_pred1_train))

print('MSE Score TEST  :', mean_squared_error(y1_test, lin_pred1_test))
# Perform linear regression on player Wage

lin_reg2 = LinearRegression(normalize=True)



run_time = time.time()

lin_reg2.fit(X2_train, y2_train)

lin_pred2_train = lin_reg2.predict(X2_train)

lin_pred2_test = lin_reg2.predict(X2_test)

run_time = time.time() - run_time



print('NAIVE LINEAR REGRESSION')

print('PREDICTING PLAYER WAGE')

print()



lin_coef = pd.Series(lin_reg2.coef_, index=X2_train.columns)

print('Most significant Coef:')

print(lin_coef.reindex(lin_coef.abs().sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y2_train, lin_pred2_train))

print('R2 Score TEST   :', r2_score(y2_test, lin_pred2_test))

print('MSE Score TRAIN :', mean_squared_error(y2_train, lin_pred2_train))

print('MSE Score TEST  :', mean_squared_error(y2_test, lin_pred2_test))
# Find best alpha for ridge regression on player Value

alphas = 10**np.linspace(-5,5,100)

params = {'alpha': alphas}



regressor = GridSearchCV(Ridge(), params, scoring='neg_mean_squared_error', cv=5)



run_time = time.time()

regressor.fit(X1_train, y1_train)

run_time = time.time() - run_time



print('RUNTIME :', run_time * 1000, 'ms')

print('Lambda  :', regressor.best_params_)

print('MSE     :', regressor.best_score_)
# Perform ridge regression on player Value

best_alpha = regressor.best_params_['alpha']



ridge_reg1 = Ridge(alpha=best_alpha, normalize=True)



run_time = time.time()

ridge_reg1.fit(X1_train, y1_train)

ridge_pred1_train = ridge_reg1.predict(X1_train)

ridge_pred1_test = ridge_reg1.predict(X1_test)

run_time = time.time() - run_time



print('NAIVE RIDGE REGRESSION')

print('PREDICTING PLAYER VALUE')

print()



ridge_coef = pd.Series(ridge_reg1.coef_, index=X1_train.columns)

print('Most significant Coef:')

print(ridge_coef.reindex(ridge_coef.abs().sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y1_train, ridge_pred1_train))

print('R2 Score TEST   :', r2_score(y1_test, ridge_pred1_test))

print('MSE Score TRAIN :', mean_squared_error(y1_train, ridge_pred1_train))

print('MSE Score TEST  :', mean_squared_error(y1_test, ridge_pred1_test))
# Find best alpha for ridge regression on player Wage

alphas = 10**np.linspace(-5,5,100)

params = {'alpha': alphas}



regressor = GridSearchCV(Ridge(), params, scoring='neg_mean_squared_error', cv=5)



run_time = time.time()

regressor.fit(X2_train, y2_train)

run_time = time.time() - run_time



print('RUNTIME :', run_time * 1000, 'ms')

print('Lambda  :', regressor.best_params_)

print('MSE     :', regressor.best_score_)
# Perform ridge regression on player Wage

best_alpha = regressor.best_params_['alpha']



ridge_reg2 = Ridge(alpha=best_alpha, normalize=True)



run_time = time.time()

ridge_reg2.fit(X2_train, y2_train)

ridge_pred2_train = ridge_reg2.predict(X2_train)

ridge_pred2_test = ridge_reg2.predict(X2_test)

run_time = time.time() - run_time



print('NAIVE RIDGE REGRESSION')

print('PREDICTING PLAYER WAGE')

print()



ridge_coef = pd.Series(ridge_reg2.coef_, index=X1_train.columns)

print('Most significant Coef:')

print(ridge_coef.reindex(ridge_coef.abs().sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y2_train, ridge_pred2_train))

print('R2 Score TEST   :', r2_score(y2_test, ridge_pred2_test))

print('MSE Score TRAIN :', mean_squared_error(y2_train, ridge_pred2_train))

print('MSE Score TEST  :', mean_squared_error(y2_test, ridge_pred2_test))
# Perform random forest regression on player Value

forest_reg1 = RandomForestRegressor(max_depth=8, random_state=42)



run_time = time.time()

forest_reg1.fit(X1_train, y1_train)

forest_pred1_train = forest_reg1.predict(X1_train)

forest_pred1_test = forest_reg1.predict(X1_test)

run_time = time.time() - run_time



print('NAIVE RANDOM FOREST REGRESSION')

print('PREDICTING PLAYER VALUE')

print()



forest_imp = pd.Series(forest_reg1.feature_importances_, index=X1_train.columns)

print('Most important Features:')

print(forest_imp.reindex(forest_imp.sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y1_train, forest_pred1_train))

print('R2 Score TEST   :', r2_score(y1_test, forest_pred1_test))

print('MSE Score TRAIN :', mean_squared_error(y1_train, forest_pred1_train))

print('MSE Score TEST  :', mean_squared_error(y1_test, forest_pred1_test))
# Perform random forest regression on player Wage

forest_reg2 = RandomForestRegressor(max_depth=8, random_state=42)



run_time = time.time()

forest_reg2.fit(X2_train, y2_train)

forest_pred2_train = forest_reg2.predict(X2_train)

forest_pred2_test = forest_reg2.predict(X2_test)

run_time = time.time() - run_time



print('NAIVE RANDOM FOREST REGRESSION')

print('PREDICTING PLAYER WAGE')

print()



forest_imp = pd.Series(forest_reg2.feature_importances_, index=X2_train.columns)

print('Most important Features:')

print(forest_imp.reindex(forest_imp.sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y2_train, forest_pred2_train))

print('R2 Score TEST   :', r2_score(y2_test, forest_pred2_test))

print('MSE Score TRAIN :', mean_squared_error(y2_train, forest_pred2_train))

print('MSE Score TEST  :', mean_squared_error(y2_test, forest_pred2_test))
fig, axs = plt.subplots(2, 3, figsize=(15,7))

fig.suptitle('Naive Regression Coeficients/Importance')



axs[0,0].plot(np.sort(np.abs(lin_reg1.coef_)))

axs[0,0].set_yscale('log')

axs[0,0].set_title('LinReg Player Value')



axs[0,1].plot(np.sort(np.abs(ridge_reg1.coef_)))

axs[0,1].set_yscale('log')

axs[0,1].set_title('RidgeReg Player Value')



axs[0,2].plot(np.sort(forest_reg1.feature_importances_))

axs[0,2].set_yscale('log')

axs[0,2].set_title('ForestReg Player Value')



axs[1,0].plot(np.sort(np.abs(lin_reg2.coef_)))

axs[1,0].set_yscale('log')

axs[1,0].set_title('LinReg Player Wage')



axs[1,1].plot(np.sort(np.abs(ridge_reg2.coef_)))

axs[1,1].set_yscale('log')

axs[1,1].set_title('RidgeReg Player Wage')



axs[1,2].plot(np.sort(forest_reg2.feature_importances_))

axs[1,2].set_yscale('log')

axs[1,2].set_title('ForestReg Player Wage')



plt.show()
# Create copy of dataset to preserve original

data_clean = data.iloc[:]
# Show unique values in Preferred foot

data['Preferred Foot'].unique()
# Assign value to left and right preferred foot

data_clean = data_clean.drop(['Preferred Foot'], axis=1, errors='ignore')

data_clean['pref_foot_val'] = data['Preferred Foot'].apply(lambda x: 0 if x == 'Left' else 1)



data_clean[['pref_foot_val']].head()
# List unique values for Body Type

data['Body Type'].unique()
# List all unique values in Body Type column

data['Body Type'].value_counts()
# Transform the abnormal unique values into their proper values

def rebodytype(b):

    if b == 'C. Ronaldo':

        return 'Lean'

    if b == 'Messi':

        return 'Normal'

    if b == 'Shaqiri':

        return 'Normal'

    if b == 'Neymar':

        return 'Normal'

    if b == 'Akinfenwa':

        return 'Stocky'

    if b == 'PLAYER_BODY_TYPE_25':

        return 'Lean'



    return b



# Clean body type values

data['Body Type'].apply(rebodytype).value_counts()
# List unique values for AWR

data['AWR'].unique()
# List unique values for DWR

data['DWR'].unique()
# Assigns a numeric value to list of ordered ordinal data

def ordinal2int(ordinal):

    return lambda x: ordinal.index(x) + 1



# Clean convert ordinal values to integer

data_clean = data_clean.drop(['Body Type', 'AWR', 'DWR'], axis=1, errors='ignore')

data_clean['body_type_val'] = data['Body Type'].apply(rebodytype).apply(ordinal2int(['Lean', 'Normal', 'Stocky']))

data_clean['awr_val'] = data['AWR'].apply(ordinal2int(['Low', 'Medium', 'High']))

data_clean['dwr_val'] = data['DWR'].apply(ordinal2int(['Low', 'Medium', 'High']))



data_clean[['body_type_val', 'awr_val', 'dwr_val']].head()
# Use value of player's specialized position

# This is more than likely his most valued attribute

data_clean['pos_rating'] = data.lookup(data.index, data['Position'])



# Simple function to get n maximum values

def nmax(arr, n=1):

    arr.sort()

    return sum(arr[-n:])



# Use sum of 3 best positions

# Player's performance will center on several positions, not all position

data_clean['flex_rating'] = np.array([nmax(d, 3) for d in data[pos_col].values])



# Drop original fields position rating fields

data_clean = data_clean.drop(pos_col, axis=1, errors='ignore')



data_clean[['pos_rating', 'flex_rating']].head()
# # Get one hot encoding for position column

# one_hot_attr = ['Position']

# one_hot_dummies = pd.get_dummies(data[one_hot_attr])



# one_hot_dummies.head()
# # Append body type encoding to dataset

# data_clean = data_clean.drop(one_hot_attr, axis=1, errors='ignore')

# data_clean = data_clean.drop(one_hot_dummies.columns, axis=1, errors='ignore')

# data_clean = data_clean.join(one_hot_dummies)



# data_clean.head()
# Split data into attributes and targets

X  = data_clean.drop(['value_val', 'wage_val'], axis=1)

y1 = data_clean['value_val']

y2 = data_clean['wage_val']



X1_train_prep, X1_test_prep, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=69)

X2_train_prep, X2_test_prep, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=69)



X1_train_prep.info()
# Create encoding for Club with mean encoding from training data

club_value_enc = X1_train_prep.join(y1_train).groupby('Club').mean()[['value_val']]

club_value_enc.columns = ['club_val']



# Create encoding for Nationality with mean encoding from training data

nat_value_enc = X1_train_prep.join(y1_train).groupby('Nationality').mean()[['value_val']]

nat_value_enc.columns = ['nationality_val']



# Create encoding for Position with mean encoding from training data

pos_value_enc = X1_train_prep.join(y1_train).groupby('Position').mean()[['value_val']]

pos_value_enc.columns = ['position_val']



value_mean = y1_train.mean()
# Encode training dataset with Club and Nationality mean encoding

X1_train = X1_train_prep.iloc[:]

X1_train = X1_train.join(X1_train.merge(club_value_enc, on='Club', right_index=True)['club_val'])

X1_train = X1_train.join(X1_train.merge(nat_value_enc, on='Nationality', right_index=True)['nationality_val'])

X1_train = X1_train.join(X1_train.merge(pos_value_enc, on='Position', right_index=True)['position_val'])



# Encode test dataset with Club and Nationality mean encoding

X1_test = X1_test_prep.iloc[:]

X1_test = X1_test.join(X1_test.merge(club_value_enc, on='Club', right_index=True)['club_val'])

X1_test = X1_test.join(X1_test.merge(nat_value_enc, on='Nationality', right_index=True)['nationality_val'])

X1_test = X1_test.join(X1_test.merge(pos_value_enc, on='Position', right_index=True)['position_val'])



# Fill out data not found in training dataset with training mean

X1_test['club_val'] = X1_test['club_val'].fillna(value_mean)

X1_test['nationality_val'] = X1_test['nationality_val'].fillna(value_mean)

X1_test['position_val'] = X1_test['position_val'].fillna(value_mean)



# Drop unencoded columns

X1_train = X1_train.drop(['Club', 'Nationality', 'Position'], axis=1, errors='ignore')

X1_test = X1_test.drop(['Club', 'Nationality', 'Position'], axis=1, errors='ignore')



X1_train[['club_val', 'nationality_val', 'position_val']].head()
# Create encoding for Club with mean encoding from training data

club_value_enc = X2_train_prep.join(y2_train).groupby('Club').mean()[['wage_val']]

club_value_enc.columns = ['club_val']



# Create encoding for Nationality with mean encoding from training data

nat_value_enc = X2_train_prep.join(y2_train).groupby('Nationality').mean()[['wage_val']]

nat_value_enc.columns = ['nationality_val']



# Create encoding for Position with mean encoding from training data

pos_value_enc = X2_train_prep.join(y2_train).groupby('Position').mean()[['wage_val']]

pos_value_enc.columns = ['position_val']



value_mean = y2_train.mean()
# Encode training dataset with Club and Nationality mean encoding

X2_train = X2_train_prep.iloc[:]

X2_train = X2_train.join(X2_train.merge(club_value_enc, on='Club', right_index=True)['club_val'])

X2_train = X2_train.join(X2_train.merge(nat_value_enc, on='Nationality', right_index=True)['nationality_val'])

X2_train = X2_train.join(X2_train.merge(pos_value_enc, on='Position', right_index=True)['position_val'])



# Encode test dataset with Club and Nationality mean encoding

X2_test = X2_test_prep.iloc[:]

X2_test = X2_test.join(X2_test.merge(club_value_enc, on='Club', right_index=True)['club_val'])

X2_test = X2_test.join(X2_test.merge(nat_value_enc, on='Nationality', right_index=True)['nationality_val'])

X2_test = X2_test.join(X2_test.merge(pos_value_enc, on='Position', right_index=True)['position_val'])



# Fill out data not found in training dataset with training mean

X2_test['club_val'] = X2_test['club_val'].fillna(value_mean)

X2_test['nationality_val'] = X2_test['nationality_val'].fillna(value_mean)

X2_test['position_val'] = X2_test['position_val'].fillna(value_mean)



# Drop unencoded columns

X2_train = X2_train.drop(['Club', 'Nationality', 'Position'], axis=1, errors='ignore')

X2_test = X2_test.drop(['Club', 'Nationality', 'Position'], axis=1, errors='ignore')



X2_train[['club_val', 'nationality_val', 'position_val']].head()
# Perform linear regression on player Value

lin_reg1 = LinearRegression(normalize=True)



run_time = time.time()

lin_reg1.fit(X1_train, y1_train)

lin_pred1_train = lin_reg1.predict(X1_train)

lin_pred1_test = lin_reg1.predict(X1_test)

run_time = time.time() - run_time



print('PREPROCESSED LINEAR REGRESSION')

print('PREDICTING PLAYER VALUE')

print()



lin_coef = pd.Series(lin_reg1.coef_, index=X1_train.columns)

print('Most significant Coef:')

print(lin_coef.reindex(lin_coef.abs().sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y1_train, lin_pred1_train))

print('R2 Score TEST   :', r2_score(y1_test, lin_pred1_test))

print('MSE Score TRAIN :', mean_squared_error(y1_train, lin_pred1_train))

print('MSE Score TEST  :', mean_squared_error(y1_test, lin_pred1_test))
# Perform linear regression on player Wage

lin_reg2 = LinearRegression(normalize=True)



run_time = time.time()

lin_reg2.fit(X2_train, y2_train)

lin_pred2_train = lin_reg2.predict(X2_train)

lin_pred2_test = lin_reg2.predict(X2_test)

run_time = time.time() - run_time



print('PREPROCESSED LINEAR REGRESSION')

print('PREDICTING PLAYER WAGE')

print()



lin_coef = pd.Series(lin_reg2.coef_, index=X2_train.columns)

print('Most significant Coef:')

print(lin_coef.reindex(lin_coef.abs().sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y2_train, lin_pred2_train))

print('R2 Score TEST   :', r2_score(y2_test, lin_pred2_test))

print('MSE Score TRAIN :', mean_squared_error(y2_train, lin_pred2_train))

print('MSE Score TEST  :', mean_squared_error(y2_test, lin_pred2_test))
# Find best alpha for ridge regression on player Value

alphas = 10**np.linspace(-5,5,100)

params = {'alpha': alphas}



regressor = GridSearchCV(Ridge(), params, scoring='neg_mean_squared_error', cv=5)



run_time = time.time()

regressor.fit(X1_train, y1_train)

run_time = time.time() - run_time



print('RUNTIME :', run_time * 1000, 'ms')

print('Lambda  :', regressor.best_params_)

print('MSE     :', regressor.best_score_)
# Perform ridge regression on player Value

best_alpha = regressor.best_params_['alpha']



ridge_reg1 = Ridge(alpha=best_alpha, normalize=True)



run_time = time.time()

ridge_reg1.fit(X1_train, y1_train)

ridge_pred1_train = ridge_reg1.predict(X1_train)

ridge_pred1_test = ridge_reg1.predict(X1_test)

run_time = time.time() - run_time



print('PREPROCESSED RIDGE REGRESSION')

print('PREDICTING PLAYER VALUE')

print()



ridge_coef = pd.Series(ridge_reg1.coef_, index=X1_train.columns)

print('Most significant Coef:')

print(ridge_coef.reindex(ridge_coef.abs().sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y1_train, ridge_pred1_train))

print('R2 Score TEST   :', r2_score(y1_test, ridge_pred1_test))

print('MSE Score TRAIN :', mean_squared_error(y1_train, ridge_pred1_train))

print('MSE Score TEST  :', mean_squared_error(y1_test, ridge_pred1_test))
# Find best alpha for ridge regression on player Wage

alphas = 10**np.linspace(-5,5,100)

params = {'alpha': alphas}



regressor = GridSearchCV(Ridge(), params, scoring='neg_mean_squared_error', cv=5)



run_time = time.time()

regressor.fit(X2_train, y2_train)

run_time = time.time() - run_time



print('RUNTIME :', run_time * 1000, 'ms')

print('Lambda  :', regressor.best_params_)

print('MSE     :', regressor.best_score_)
# Perform ridge regression on player Wage

best_alpha = regressor.best_params_['alpha']



ridge_reg2 = Ridge(alpha=best_alpha, normalize=True)



run_time = time.time()

ridge_reg2.fit(X2_train, y2_train)

ridge_pred2_train = ridge_reg2.predict(X2_train)

ridge_pred2_test = ridge_reg2.predict(X2_test)

run_time = time.time() - run_time



print('PREPROCESSED RIDGE REGRESSION')

print('PREDICTING PLAYER WAGE')

print()



ridge_coef = pd.Series(ridge_reg2.coef_, index=X2_train.columns)

print('Most significant Coef:')

print(ridge_coef.reindex(ridge_coef.abs().sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y2_train, ridge_pred2_train))

print('R2 Score TEST   :', r2_score(y2_test, ridge_pred2_test))

print('MSE Score TRAIN :', mean_squared_error(y2_train, ridge_pred2_train))

print('MSE Score TEST  :', mean_squared_error(y2_test, ridge_pred2_test))
# Perform random forest regression on player Value

forest_reg1 = RandomForestRegressor(max_depth=8, random_state=42)



run_time = time.time()

forest_reg1.fit(X1_train, y1_train)

forest_pred1_train = forest_reg1.predict(X1_train)

forest_pred1_test = forest_reg1.predict(X1_test)

run_time = time.time() - run_time



print('PREPROCESSED RANDOM FOREST REGRESSION')

print('PREDICTING PLAYER VALUE')

print()



forest_imp = pd.Series(forest_reg1.feature_importances_, index=X1_train.columns)

print('Most important Features:')

print(forest_imp.reindex(forest_imp.sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y1_train, forest_pred1_train))

print('R2 Score TEST   :', r2_score(y1_test, forest_pred1_test))

print('MSE Score TRAIN :', mean_squared_error(y1_train, forest_pred1_train))

print('MSE Score TEST  :', mean_squared_error(y1_test, forest_pred1_test))
# Perform random forest regression on player Wage

forest_reg2 = RandomForestRegressor(max_depth=8, random_state=42)



run_time = time.time()

forest_reg2.fit(X2_train, y2_train)

forest_pred2_train = forest_reg2.predict(X2_train)

forest_pred2_test = forest_reg2.predict(X2_test)

run_time = time.time() - run_time



print('PREPROCESSED RANDOM FOREST REGRESSION')

print('PREDICTING PLAYER WAGE')

print()



forest_imp = pd.Series(forest_reg2.feature_importances_, index=X2_train.columns)

print('Most important Features:')

print(forest_imp.reindex(forest_imp.sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y2_train, forest_pred2_train))

print('R2 Score TEST   :', r2_score(y2_test, forest_pred2_test))

print('MSE Score TRAIN :', mean_squared_error(y2_train, forest_pred2_train))

print('MSE Score TEST  :', mean_squared_error(y2_test, forest_pred2_test))
fig, axs = plt.subplots(2, 3, figsize=(15,7))

fig.suptitle('Pre-processed Regression Coeficients/Importance')



axs[0,0].plot(np.sort(np.abs(lin_reg1.coef_)))

axs[0,0].set_yscale('log')

axs[0,0].set_title('LinReg Player Value')



axs[0,1].plot(np.sort(np.abs(ridge_reg1.coef_)))

axs[0,1].set_yscale('log')

axs[0,1].set_title('RidgeReg Player Value')



axs[0,2].plot(np.sort(forest_reg1.feature_importances_))

axs[0,2].set_yscale('log')

axs[0,2].set_title('ForestReg Player Value')



axs[1,0].plot(np.sort(np.abs(lin_reg2.coef_)))

axs[1,0].set_yscale('log')

axs[1,0].set_title('LinReg Player Wage')



axs[1,1].plot(np.sort(np.abs(ridge_reg2.coef_)))

axs[1,1].set_yscale('log')

axs[1,1].set_title('RidgeReg Player Wage')



axs[1,2].plot(np.sort(forest_reg2.feature_importances_))

axs[1,2].set_yscale('log')

axs[1,2].set_title('ForestReg Player Wage')



plt.show()
X1_train.shape
# Sanity check whether PCA is worth it

pca = PCA().fit(X1_train)

pd.DataFrame(pca.transform(X1_train)).shape
# Get covariance matrix

c = X1_train.cov().to_numpy()

eigval, eigvec = np.linalg.eig(c)



# Get sorted eigen values and vectors

idx = eigval.argsort()[::-1]

eigval = eigval[idx]

eigvec = eigvec[:,idx]



eigval
# Plot out eigen values to see where to cut off

plt.plot(eigval)

plt.yscale('log')

plt.show()
# Cut off eigen vector at 52 attributes since there doesn't seem to be a good cutoff point

eigmat = eigvec[:,:52]

eigmat
# Generate test and train dataset transformed with PCA

X1_train_pca = X1_train.dot(eigmat)

X1_test_pca = X1_test.dot(eigmat)



X1_train_pca.head()
# Check reconstruction error

X1_pca_recon = X1_train_pca.dot(eigmat.transpose()).to_numpy()

X1_original = X1_train.to_numpy()

((X1_pca_recon - X1_original)**2).mean()
def x1_pca_recon_err(n):

    eigmat = eigvec[:,:n]



    X1_train_pca = X1_train.dot(eigmat)

    X1_pca_recon = X1_train_pca.dot(eigmat.transpose()).to_numpy()

    X1_original = X1_train.to_numpy()

    return ((X1_pca_recon - X1_original)**2).mean()



for i in range(53, 33, -1):

    print('attributes:',  i, '        reconstruction err:', x1_pca_recon_err(i))
# Get covariance matrix

c = X2_train.cov().to_numpy()

eigval, eigvec = np.linalg.eig(c)



# Get sorted eigen values and vectors

idx = eigval.argsort()[::-1]

eigval = eigval[idx]

eigvec = eigvec[:,idx]



eigval
def x2_pca_recon_err(n):

    eigmat = eigvec[:,:n]



    X2_train_pca = X2_train.dot(eigmat)

    X2_pca_recon = X2_train_pca.dot(eigmat.transpose()).to_numpy()

    X2_original = X2_train.to_numpy()

    return ((X2_pca_recon - X2_original)**2).mean()



for i in range(53, 33, -1):

    print('attributes:',  i, '        reconstruction err:', x2_pca_recon_err(i))
# Cut off eigen vector at 52 attributes since there doesn't seem to be a good cutoff point

eigmat = eigvec[:,:52]



# Generate test and train dataset transformed with PCA

X2_train_pca = X2_train.dot(eigmat)

X2_test_pca = X2_test.dot(eigmat)



X2_train_pca.head()
# Perform linear regression on player Value

lin_reg1 = LinearRegression(normalize=True)



run_time = time.time()

lin_reg1.fit(X1_train_pca, y1_train)

lin_pred1_train = lin_reg1.predict(X1_train_pca)

lin_pred1_test = lin_reg1.predict(X1_test_pca)

run_time = time.time() - run_time



print('PCA LINEAR REGRESSION')

print('PREDICTING PLAYER VALUE')

print()



lin_coef = pd.Series(lin_reg1.coef_, index=X1_train_pca.columns)

print('Most significant Coef:')

print(lin_coef.reindex(lin_coef.abs().sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y1_train, lin_pred1_train))

print('R2 Score TEST   :', r2_score(y1_test, lin_pred1_test))

print('MSE Score TRAIN :', mean_squared_error(y1_train, lin_pred1_train))

print('MSE Score TEST  :', mean_squared_error(y1_test, lin_pred1_test))
# Perform linear regression on player Wage

lin_reg2 = LinearRegression(normalize=True)



run_time = time.time()

lin_reg2.fit(X2_train_pca, y2_train)

lin_pred2_train = lin_reg2.predict(X2_train_pca)

lin_pred2_test = lin_reg2.predict(X2_test_pca)

run_time = time.time() - run_time



print('PCA LINEAR REGRESSION')

print('PREDICTING PLAYER WAGE')

print()



lin_coef = pd.Series(lin_reg2.coef_, index=X2_train_pca.columns)

print('Most significant Coef:')

print(lin_coef.reindex(lin_coef.abs().sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y2_train, lin_pred2_train))

print('R2 Score TEST   :', r2_score(y2_test, lin_pred2_test))

print('MSE Score TRAIN :', mean_squared_error(y2_train, lin_pred2_train))

print('MSE Score TEST  :', mean_squared_error(y2_test, lin_pred2_test))
# Find best alpha for ridge regression on player Value

alphas = 10**np.linspace(-5,5,100)

params = {'alpha': alphas}



regressor = GridSearchCV(Ridge(), params, scoring='neg_mean_squared_error', cv=5)



run_time = time.time()

regressor.fit(X1_train_pca, y1_train)

run_time = time.time() - run_time



print('RUNTIME :', run_time * 1000, 'ms')

print('Lambda  :', regressor.best_params_)

print('MSE     :', regressor.best_score_)
# Perform ridge regression on player Value

best_alpha = regressor.best_params_['alpha']



ridge_reg1 = Ridge(alpha=best_alpha, normalize=True)



run_time = time.time()

ridge_reg1.fit(X1_train_pca, y1_train)

ridge_pred1_train = ridge_reg1.predict(X1_train_pca)

ridge_pred1_test = ridge_reg1.predict(X1_test_pca)

run_time = time.time() - run_time



print('PCA RIDGE REGRESSION')

print('PREDICTING PLAYER VALUE')

print()



ridge_coef = pd.Series(ridge_reg1.coef_, index=X1_train_pca.columns)

print('Most significant Coef:')

print(ridge_coef.reindex(ridge_coef.abs().sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y1_train, ridge_pred1_train))

print('R2 Score TEST   :', r2_score(y1_test, ridge_pred1_test))

print('MSE Score TRAIN :', mean_squared_error(y1_train, ridge_pred1_train))

print('MSE Score TEST  :', mean_squared_error(y1_test, ridge_pred1_test))
# Find best alpha for ridge regression on player Wage

alphas = 10**np.linspace(-5,5,100)

params = {'alpha': alphas}



regressor = GridSearchCV(Ridge(), params, scoring='neg_mean_squared_error', cv=5)



run_time = time.time()

regressor.fit(X2_train_pca, y2_train)

run_time = time.time() - run_time



print('RUNTIME :', run_time * 1000, 'ms')

print('Lambda  :', regressor.best_params_)

print('MSE     :', regressor.best_score_)
# Perform ridge regression on player Wage

best_alpha = regressor.best_params_['alpha']



ridge_reg2 = Ridge(alpha=best_alpha, normalize=True)



run_time = time.time()

ridge_reg2.fit(X2_train_pca, y2_train)

ridge_pred2_train = ridge_reg2.predict(X2_train_pca)

ridge_pred2_test = ridge_reg2.predict(X2_test_pca)

run_time = time.time() - run_time



print('PCA RIDGE REGRESSION')

print('PREDICTING PLAYER WAGE')

print()



ridge_coef = pd.Series(ridge_reg2.coef_, index=X1_train_pca.columns)

print('Most significant Coef:')

print(ridge_coef.reindex(ridge_coef.abs().sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y2_train, ridge_pred2_train))

print('R2 Score TEST   :', r2_score(y2_test, ridge_pred2_test))

print('MSE Score TRAIN :', mean_squared_error(y2_train, ridge_pred2_train))

print('MSE Score TEST  :', mean_squared_error(y2_test, ridge_pred2_test))
# Perform random forest regression on player Value

forest_reg1 = RandomForestRegressor(max_depth=8, random_state=42)



run_time = time.time()

forest_reg1.fit(X1_train_pca, y1_train)

forest_pred1_train = forest_reg1.predict(X1_train_pca)

forest_pred1_test = forest_reg1.predict(X1_test_pca)

run_time = time.time() - run_time



print('PCA RANDOM FOREST REGRESSION')

print('PREDICTING PLAYER VALUE')

print()



forest_imp = pd.Series(forest_reg1.feature_importances_, index=X1_train_pca.columns)

print('Most important Features:')

print(forest_imp.reindex(forest_imp.sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y1_train, forest_pred1_train))

print('R2 Score TEST   :', r2_score(y1_test, forest_pred1_test))

print('MSE Score TRAIN :', mean_squared_error(y1_train, forest_pred1_train))

print('MSE Score TEST  :', mean_squared_error(y1_test, forest_pred1_test))
# Perform random forest regression on player Wage

forest_reg2 = RandomForestRegressor(max_depth=8, random_state=42)



run_time = time.time()

forest_reg2.fit(X2_train_pca, y2_train)

forest_pred2_train = forest_reg2.predict(X2_train_pca)

forest_pred2_test = forest_reg2.predict(X2_test_pca)

run_time = time.time() - run_time



print('PCA RANDOM FOREST REGRESSION')

print('PREDICTING PLAYER WAGE')

print()



forest_imp = pd.Series(forest_reg2.feature_importances_, index=X2_train_pca.columns)

print('Most important Features:')

print(forest_imp.reindex(forest_imp.sort_values(ascending=False).index).head(10))

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y2_train, forest_pred2_train))

print('R2 Score TEST   :', r2_score(y2_test, forest_pred2_test))

print('MSE Score TRAIN :', mean_squared_error(y2_train, forest_pred2_train))

print('MSE Score TEST  :', mean_squared_error(y2_test, forest_pred2_test))
fig, axs = plt.subplots(2, 3, figsize=(15,7))

fig.suptitle('PCA Regression Coeficients/Importance')



axs[0,0].plot(np.sort(np.abs(lin_reg1.coef_)))

axs[0,0].set_yscale('log')

axs[0,0].set_title('LinReg Player Value')



axs[0,1].plot(np.sort(np.abs(ridge_reg1.coef_)))

axs[0,1].set_yscale('log')

axs[0,1].set_title('RidgeReg Player Value')



axs[0,2].plot(np.sort(forest_reg1.feature_importances_))

axs[0,2].set_yscale('log')

axs[0,2].set_title('ForestReg Player Value')



axs[1,0].plot(np.sort(np.abs(lin_reg2.coef_)))

axs[1,0].set_yscale('log')

axs[1,0].set_title('LinReg Player Wage')



axs[1,1].plot(np.sort(np.abs(ridge_reg2.coef_)))

axs[1,1].set_yscale('log')

axs[1,1].set_title('RidgeReg Player Wage')



axs[1,2].plot(np.sort(forest_reg2.feature_importances_))

axs[1,2].set_yscale('log')

axs[1,2].set_title('ForestReg Player Wage')



plt.show()
X1_train.nunique().sum()
X1_train_pca.nunique().sum()
# Split data into attributes and targets

X  = data.drop(['value_val', 'wage_val'], axis=1)

y1 = data['value_val']

y2 = data['wage_val']



X1_train, X1_test , y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42)

X2_train, X2_test , y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=42)



X2_train.info()
# Perform slope boost regression with prebuilt model on player Value

cat_features = [1, 4, 6, 10, 11, 46, 47]

catboost_reg1 = CatBoostRegressor(iterations=200,

                                 learning_rate=0.1,

                                 depth=8)



run_time = time.time()

catboost_reg1.fit(X1_train, y1_train, cat_features, verbose=False)

catboost_pred1_train = catboost_reg1.predict(X1_train)

catboost_pred1_test = catboost_reg1.predict(X1_test)

run_time = time.time() - run_time

print()



print('CLEAN CATBOOST REGRESSION')

print('PREDICTING PLAYER VALUE')

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y1_train, catboost_pred1_train))

print('R2 Score TEST   :', r2_score(y1_test, catboost_pred1_test))

print('MSE Score TRAIN :', mean_squared_error(y1_train, catboost_pred1_train))

print('MSE Score TEST  :', mean_squared_error(y1_test, catboost_pred1_test))
# Perform slope boost regression with prebuilt model on player Wage

cat_features = [1, 4, 6, 10, 11, 46, 47]

catboost_reg2 = CatBoostRegressor(iterations=200,

                                 learning_rate=0.1,

                                 depth=8)



run_time = time.time()

catboost_reg2.fit(X2_train, y2_train, cat_features, verbose=False)

catboost_pred2_train = catboost_reg2.predict(X2_train)

catboost_pred2_test = catboost_reg2.predict(X2_test)

run_time = time.time() - run_time

print()



print('CLEAN CATBOOST REGRESSION')

print('PREDICTING PLAYER WAGE')

print()



print('RUNTIME         :', run_time * 1000, 'ms')

print('R2 Score TRAIN  :', r2_score(y2_train, catboost_pred2_train))

print('R2 Score TEST   :', r2_score(y2_test, catboost_pred2_test))

print('MSE Score TRAIN :', mean_squared_error(y2_train, catboost_pred2_train))

print('MSE Score TEST  :', mean_squared_error(y2_test, catboost_pred2_test))
# Previous runtime results

print("DISCLAIMER: These results are taken from a previous run, and thus may not fully reflect the results of the current run")



pd.DataFrame(

    index=['Naive', 'Preprocess', 'PCA', 'Catboost'],

    data={

        'LINREG_RUNTIME' : [

            1395.0705528259277,

            43.75290870666504,

            31.097888946533203,

            None,

        ],

        'RIDGEREG_RUNTIME' : [

            481.5702438354492,

            30.005216598510742,

            26.578664779663086,

            None,

        ],

        'FORESTREG_RUNTIME' : [

            49870.77331542969,

            14616.775751113892,

            37502.03585624695,

            None,

        ],

        'CATBOOST_RUNTIME' : [

            None,

            None,

            None,

            5879.0154457092285,

        ],

    }

)
# Previous R2 results

print("DISCLAIMER: These results are taken from a previous run, and thus may not fully reflect the results of the current run")



pd.DataFrame(

    index=['Naive', 'Preprocess', 'PCA', 'Catboost'],

    data={

        'LINREG_R2_TRAIN' : [

            0.7193842683071934,

            0.6811962636493403,

            0.6764538606726269,

            None,

        ],

        'LINREG_R2_TEST' : [

            -2.007368207784571e+24,

            0.6908466302454932,

            0.687966745331998,

            None,

        ],

        'RIDGEREG_R2_TRAIN' : [

            0.747300305671266,

            0.4362093255187618,

            0.08442516570869185,

            None,

        ],

        'RIDGEREG_R2_TEST' : [

            0.6922009795979305,

            0.45298678097485645,

            0.0845167641060749,

            None,

        ],

        'FORESTREG_R2_TRAIN' : [

            0.9958668452361841,

            0.9952358572702557,

            0.9230677146733887,

            None,

        ],

        'FORESTREG_R2_TEST' : [

            0.968032534257517,

            0.983041207042161,

            0.753495776123223,

            None,

        ],

        'CATBOOST_R2_TRAIN' : [

            None,

            None,

            None,

            0.9990151544521757,

        ],

        'CATBOOST_R2_TEST' : [

            None,

            None,

            None,

            0.9839465566697981,

        ],

    }

)
# Previous MSE results

print("DISCLAIMER: These results are taken from a previous run, and thus may not fully reflect the results of the current run")



pd.DataFrame(

    index=['Naive', 'Preprocess', 'PCA', 'Catboost'],

    data={

        'LINREG_MSE_TRAIN' : [

            9190175287380.48,

            10255749711879.357,

            10408310339050.133,

            None,

        ],

        'LINREG_MSE_TEST' : [

            6.766706098142399e+37,

            11130832118237.354,

            11234520185805.291,

            None,

        ],

        'RIDGEREG_MSE_TRAIN' : [

            8275924061487.902,

            18136851573820.21,

            29453564285265.76,

            None,

        ],

        'RIDGEREG_MSE_TEST' : [

            10375702376275.074,

            19694795215269.402,

            32961278131588.008,

            None,

        ],

        'FORESTREG_MSE_TRAIN' : [

            135360966900.18025,

            153259983672.79407,

            2474871443177.7456,

            None,

        ],

        'FORESTREG_MSE_TEST' : [

            1077602228345.3073,

            610588451588.1108,

            8875197234911.596,

            None,

        ],

        'CATBOOST_MSE_TRAIN' : [

            None,

            None,

            None,

            30815427200.89645,

        ],

        'CATBOOST_MSE_TEST' : [

            None,

            None,

            None,

            634936722810.1262,

        ],

    }

)
# Previous runtime results

print("DISCLAIMER: These results are taken from a previous run, and thus may not fully reflect the results of the current run")



pd.DataFrame(

    index=['Naive', 'Preprocess', 'PCA', 'Catboost'],

    data={

        'LINREG_RUNTIME' : [

            1298.584222793579,

            30.440807342529297,

            29.247760772705078,

            None,

        ],

        'RIDGEREG_RUNTIME' : [

            559.8382949829102,

            28.866291046142578,

            27.935028076171875,

            None,

        ],

        'FORESTREG_RUNTIME' : [

            48230.66830635071,

            14755.709409713745,

            38085.474729537964,

            None,

        ],

        'CATBOOST_RUNTIME' : [

            None,

            None,

            None,

            6367.7990436553955,

        ],

    }

)
# Previous R2 results

print("DISCLAIMER: These results are taken from a previous run, and thus may not fully reflect the results of the current run")



pd.DataFrame(

    index=['Naive', 'Preprocess', 'PCA', 'Catboost'],

    data={

        'LINREG_R2_TRAIN' : [

            0.7961566934117595,

            0.7291388329945518,

            0.7272799630194423,

            None,

        ],

        'LINREG_R2_TEST' : [

            -1.4937935438398293e+24,

            0.7591790115899942,

            0.7574900343482656,

            None,

        ],

        'RIDGEREG_R2_TRAIN' : [

            0.786210504976484,

            0.5373330779735297,

            0.3780338691031372,

            None,

        ],

        'RIDGEREG_R2_TEST' : [

            0.7574142187993411,

            0.5301747053026842,

            0.3676874489739226,

            None,

        ],

        'FORESTREG_R2_TRAIN' : [

            0.8979487218767636,

            0.9753495018510526,

            0.9593433749381992,

            None,

        ],

        'FORESTREG_R2_TEST' : [

            0.8266111288857395,

            0.9470141207397708,

            0.8938425415401196,

            None,

        ],

        'CATBOOST_R2_TRAIN' : [

            None,

            None,

            None,

            0.9664667862347623,

        ],

        'CATBOOST_R2_TEST' : [

            None,

            None,

            None,

            0.9047771928538145,

        ],

    }

)
# Previous MSE results

print("DISCLAIMER: These results are taken from a previous run, and thus may not fully reflect the results of the current run")



pd.DataFrame(

    index=['Naive', 'Preprocess', 'PCA', 'Catboost'],

    data={

        'LINREG_MSE_TRAIN' : [

            103397445.66153847,

            129496441.17487168,

            130385151.24374124,

            None,

        ],

        'LINREG_MSE_TEST' : [

            8.150553267490191e+32,

            159286113.64080223,

            160403253.06722423,

            None,

        ],

        'RIDGEREG_MSE_TRAIN' : [

            108442548.66486241,

            221197156.1451429,

            297356765.3602706,

            None,

        ],

        'RIDGEREG_MSE_TEST' : [

            132361552.8909861,

            310756324.7563297,

            418230194.6530032,

            None,

        ],

        'FORESTREG_MSE_TRAIN' : [

            51764473.70799574,

            11785195.414934598,

            19437589.795138095,

            None,

        ],

        'FORESTREG_MSE_TEST' : [

            94605793.13886109,

            35046425.317520335,

            70215678.05905038,

            None,

        ],

        'CATBOOST_MSE_TRAIN' : [

            None,

            None,

            None,

            16338483.178480953,

        ],

        'CATBOOST_MSE_TEST' : [

            None,

            None,

            None,

            59566791.78955186,

        ],

    }

)