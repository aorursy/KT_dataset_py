# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# all Seattle related data, named with "_sl", all Boston related data, named with "_bl" 
df_sl = pd.read_csv("/kaggle/input/seattle/listings.csv")
df_bl = pd.read_csv("/kaggle/input/boston/listings.csv")

num_rows_s = df_sl.shape[0]
num_cols_s = df_sl.shape[1]

num_rows_b = df_sl.shape[0]
num_cols_b = df_sl.shape[1]

most_missing_cols_s = set(df_sl.columns[df_sl.isnull().mean() > 0.75])
most_missing_cols_b = set(df_bl.columns[df_bl.isnull().mean() > 0.75])

print(num_rows_s, num_cols_s, num_rows_b, num_cols_b)
print(most_missing_cols_s, most_missing_cols_b)
df_sl.columns
df_bl.columns
# Basic Data Cleaning function for Seattle
def clean_dataset(df):
    '''
    INPUT
    df - pandas dataframe containing data 
    
    OUTPUT
    new_df - cleaned dataset, which contains:
    1. string containing price are converted into numbers;
    2. missing values are imputed with mean or mode or drop
    '''
    
    useless_columns = ['access', 'interaction', 'house_rules','name', 'host_name', 'square_feet', 'id', 'host_id','summary', 'space', 'description', 'neighborhood_overview', 'notes', 
                       'host_since', 'host_location', 'host_about', 'host_neighbourhood', 'host_total_listings_count', 'street', 'neighbourhood', 
                       'minimum_nights', 'maximum_nights', 'city', 'zipcode', 'smart_location', 'latitude', 
                       'longitude', 'is_location_exact', 'weekly_price', 'monthly_price', 'require_guest_profile_picture', 
                       'require_guest_phone_verification', 'calculated_host_listings_count', 'availability_30', 'availability_60', 'availability_90', 
                       'availability_365', 'calendar_updated','transit']
    
    # if all values are unique in this column, like ID, or if the values are url links, then drop it
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)
        if ('url' in col):
            df.drop(col, inplace=True, axis=1)
        if col in useless_columns:
            df.drop(col, inplace=True, axis=1)
    
    # generate review columns
    review_columns = []
    for col in df:
        if 'review' in col:
            review_columns.append(col)
    
    
    #convert all related 'price' columns values from string to number
    df['price'] = df['price'].str.replace("[$, ]", "").astype("float")
    df['security_deposit'] = df['security_deposit'].str.replace("[$, ]", "").astype("float")
    df['cleaning_fee'] = df['cleaning_fee'].str.replace("[$, ]", "").astype("float")
    df['extra_people'] = df['extra_people'].str.replace("[$, ]", "").astype("float")
    #convert all percentage columns values to float number
    df['host_response_rate'] = df['host_response_rate'].str.replace("[%, ]", "").astype("float")/100
    df['host_acceptance_rate'] = df['host_acceptance_rate'].str.replace("[%, ]", "").astype("float")/100
    #generate new review metric
    df['new_review_metric'] = df['reviews_per_month'] * df['review_scores_rating']/100
    #drop original review columns
    df = df.drop(review_columns, axis=1)
    
    return df
# Apply data cleaning functions above to clean dataset
clen_df_sl = clean_dataset(df_sl)
clen_df_bl = clean_dataset(df_bl)
# 'neighbourhood_group_cleansed' and 'state' are all null in Boston dataset, so we need to drop these two columns in Seattle dataset manually
clen_df_sl.drop('neighbourhood_group_cleansed', axis=1, inplace = True)
clen_df_sl.drop('state', axis=1, inplace = True)
clen_df_bl.drop('market', axis=1, inplace = True)
clen_df_sl.columns
clen_df_bl.columns
# Preprocessing the complicated multi-catigories data
#cat_df_sl = clen_df_sl.copy()

def element_len(df, colname):
    coliloc = df.columns.get_loc(colname)
    
    for i, row in enumerate(df[colname]):
        df.iloc[i, coliloc] = row.replace('[', '').replace("'", '').replace("]", '').replace('"', '').replace('{', '').replace('}', '').replace(' ','')
        df.iloc[i, coliloc] = len(df.iloc[i, coliloc].split(','))
    return df

def create_dummy_df(df, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating 
    '''
    # Dummy the categorical variables
    cat_cols = ['host_response_time', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable', 'cancellation_policy']

    for col in  cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)
        except:
            continue
    return df

clen_df_sl = element_len(clen_df_sl, 'amenities')
clen_df_sl = element_len(clen_df_sl, 'host_verifications')
clen_df_bl = element_len(clen_df_bl, 'amenities')
clen_df_bl = element_len(clen_df_bl, 'host_verifications')
clen_df_sl.describe()
clen_df_bl.describe()
clen_df_sl = create_dummy_df(clen_df_sl, dummy_na=False)
clen_df_bl = create_dummy_df(clen_df_bl, dummy_na=False)
len(clen_df_sl.columns)
len(clen_df_bl.columns)
for col in clen_df_bl:
    if col not in clen_df_sl:
        print(col)
cpsl=sns.catplot(x='property_type', y='price', kind='bar', data=clen_df_sl)
cpsl.set_xticklabels(rotation=45, horizontalalignment='right')
cpbl=sns.catplot(x='property_type', y='price', kind='bar', data=clen_df_bl)
cpbl.set_xticklabels(rotation=45, horizontalalignment='right')
cbrpl=sns.catplot(x='bed_type', y='price', col = 'room_type', kind='bar', data=clen_df_sl)
cbrpl.set_xticklabels(rotation=45, horizontalalignment='right')
cbrbl=sns.catplot(x='bed_type', y='price', col = 'room_type', kind='bar', data=clen_df_bl)
cbrbl.set_xticklabels(rotation=45, horizontalalignment='right')
cbsl=sns.catplot(x='price', y='beds', orient ='h', kind='bar',data=clen_df_sl)
cbsl.set_xticklabels(rotation=45, horizontalalignment='right')
cbbl=sns.catplot(x='price', y='beds', orient ='h', kind='bar',data=clen_df_bl)
cbbl.set_xticklabels(rotation=45, horizontalalignment='right')
cbasl=sns.catplot(x='price', y='bathrooms', kind='bar', orient = 'h', data=clen_df_sl)
cbasl.set_xticklabels(rotation=45, horizontalalignment='right')
cbabl=sns.catplot(x='price', y='bathrooms', kind='bar', orient = 'h', data=clen_df_bl)
cbabl.set_xticklabels(rotation=45, horizontalalignment='right')
casl=sns.catplot(x='price', y='accommodates', orient = 'h', kind='bar',data=clen_df_sl)
casl.set_xticklabels(rotation=45, horizontalalignment='right')
cabl=sns.catplot(x='price', y='accommodates', orient = 'h', kind='bar',data=clen_df_bl)
cabl.set_xticklabels(rotation=45, horizontalalignment='right')
cbesl=sns.catplot(x='price', y='bedrooms', orient = 'h', kind='bar',data=clen_df_sl)
cbesl.set_xticklabels(rotation=45, horizontalalignment='right')
cbebl=sns.catplot(x='price', y='bedrooms', orient = 'h', kind='bar',data=clen_df_bl)
cbebl.set_xticklabels(rotation=45, horizontalalignment='right')
# Generata new behavior_review dataframe for analysis
behavior_review_bl_cols =  ['host_response_rate', 'host_acceptance_rate',
                        'host_response_time_within a day',
                        'host_response_time_within a few hours',
                        'host_response_time_within an hour',
                        'host_has_profile_pic_t', 
                        'host_identity_verified_t', 
                        'host_is_superhost_t', 
                        'instant_bookable_t', 
                        'cancellation_policy_moderate',
                        'cancellation_policy_strict',
                        'cancellation_policy_super_strict_30',
                        'amenities',
                        'host_verifications',
                        'guests_included', 'extra_people', 'price',
                        'new_review_metric']

behavior_review_sl_cols = behavior_review_bl_cols.copy()
behavior_review_sl_cols.remove('cancellation_policy_super_strict_30')

behavior_review_sl = clen_df_sl[behavior_review_sl_cols].copy()
behavior_review_bl = clen_df_bl[behavior_review_bl_cols].copy()
corr = behavior_review_sl.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
plt.rcParams['figure.figsize'] = [11, 9]
sns.heatmap(corr, mask=mask, annot = True, fmt='.2f')
corr = behavior_review_bl.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
plt.rcParams['figure.figsize'] = [11, 9]
sns.heatmap(corr, mask=mask, annot = True, fmt=".2f")
# copy the cleaned dataset
review_df_sl= clen_df_sl.copy()
review_df_bl= clen_df_bl.copy()
def fin_clean_data(df):
    '''
    INPUT
    df - pandas dataframe 
    
    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector
    
    Perform to obtain the correct X and y objects
    This function cleans df using the following steps to produce X and y:
    1. Drop all the rows with no salaries
    2. Create X as all the columns that are not the Salary column
    3. Create y as the Salary column
    4. Drop the Salary, Respondent, and the ExpectedSalary columns from X
    5. For each numeric variable in X, fill the column with the mean value of the column.
    6. Create dummy columns for all the categorical variables in X, drop the original columns
    '''
    # drop irrelavent variables
    irrelavent_cols = ['cleaning_fee', 'security_deposit', 'host_verifications']
    
    for col in  irrelavent_cols:
        # for each cat add dummy var, drop original column
        df = df.drop(col, axis=1)
    
    # Drop rows with missing salary values
    df = df.dropna(subset=['new_review_metric'], axis=0)
    y = df['new_review_metric']
    
    # Fill numeric columns with the mean
    num_vars = df.select_dtypes(include=['float', 'int']).columns
    for col in num_vars:
        df[col].fillna((df[col].mean()), inplace=True)
        


    # Dummy the categorical variables
    cat_vars = df.select_dtypes(include=['object']).copy().columns
    for var in  cat_vars:
    #    # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)
    

    
    X = df.drop(['new_review_metric'], axis=1)
    #X = df['host_acceptance_rate'].values

    return X, y
#Use the above function to finalize the data preprocessing for X and y
X_sl, y_sl = fin_clean_data(review_df_sl)
X_bl, y_bl = fin_clean_data(review_df_bl)
#Split into train and test
X_train_sl, X_test_sl, y_train_sl, y_test_sl = train_test_split(X_sl, y_sl, test_size=0.3, random_state=42)
X_train_bl, X_test_bl, y_train_bl, y_test_bl = train_test_split(X_bl, y_bl, test_size=0.3, random_state=42)

sc = StandardScaler()
X_train_sl = sc.fit_transform(X_train_sl)
X_test_sl = sc.transform(X_test_sl)
X_train_bl = sc.fit_transform(X_train_bl)
X_test_bl = sc.transform(X_test_bl)
regressor_sl = RandomForestRegressor(n_estimators=100, 
                               criterion='mse', 
                               random_state=42, 
                               n_jobs=-1)
regressor_bl = RandomForestRegressor(n_estimators=100, 
                               criterion='mse', 
                               random_state=42, 
                               n_jobs=-1)
regressor_sl.fit(X_train_sl, y_train_sl.squeeze())
regressor_bl.fit(X_train_bl, y_train_bl.squeeze())
y_train_sl_preds = regressor_sl.predict(X_train_sl)
y_test_sl_preds = regressor_sl.predict(X_test_sl)

print('Random Forest MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train_sl, y_train_sl_preds),
        mean_squared_error(y_test_sl, y_test_sl_preds)))
print('Random Forest R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train_sl, y_train_sl_preds),
        r2_score(y_test_sl, y_test_sl_preds)))
y_train_bl_preds = regressor_bl.predict(X_train_bl)
y_test_bl_preds = regressor_bl.predict(X_test_bl)

print('Random Forest MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train_bl, y_train_bl_preds),
        mean_squared_error(y_test_bl, y_test_bl_preds)))
print('Random Forest R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train_bl, y_train_bl_preds),
        r2_score(y_test_bl, y_test_bl_preds)))
importances = regressor_sl.feature_importances_
feat_names = X_sl.columns
tree_result = pd.DataFrame({'feature': feat_names, 'importance': importances})
tree_result_sort = tree_result.sort_values(by='importance',ascending=False)[:10]
chart = sns.catplot(x='feature', y='importance', kind='bar', data=tree_result_sort)
chart.set_xticklabels(rotation=45, horizontalalignment='right')
#chart.set_titles("Seattle's feature importances analysis for reviews")
importances = regressor_bl.feature_importances_
feat_names = X_bl.columns
tree_result = pd.DataFrame({'feature': feat_names, 'importance': importances})
tree_result_sort = tree_result.sort_values(by='importance',ascending=False)[:10]
chart = sns.catplot(x='feature', y='importance', kind='bar', data=tree_result_sort)
chart.set_xticklabels(rotation=45, horizontalalignment='right')
#chart.set_titles("Boston's feature importances analysis for reviews")