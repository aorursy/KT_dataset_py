import pandas as pd
import numpy as np
import sys

def return_size(df):
    """Return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)

def convert_types(df):
    print(f'Original size of data: {return_size(df)} gb.')
    for c in df:
        if df[c].dtype == 'object':
            df[c] = df[c].astype('category')
    print(f'New size of data: {return_size(df)} gb.')
    return df
# Read in the datasets and replace the anomalous values
app_train = pd.read_csv('../input/application_train.csv').replace({365243: np.nan})
app_test = pd.read_csv('../input/application_test.csv').replace({365243: np.nan})
bureau = pd.read_csv('../input/bureau.csv').replace({365243: np.nan})
bureau_balance = pd.read_csv('../input/bureau_balance.csv').replace({365243: np.nan})

app_test['TARGET'] = np.nan
app = app_train.append(app_test, ignore_index = True, sort = True)

app = convert_types(app)
bureau = convert_types(bureau)
bureau_balance = convert_types(bureau_balance)

import gc
gc.enable()
del app_train, app_test
gc.collect()
def agg_numeric(df, parent_var, df_name):
    """
    Groups and aggregates the numeric values in a child dataframe
    by the parent variable.
    
    Parameters
    --------
        df (dataframe): 
            the child dataframe to calculate the statistics on
        parent_var (string): 
            the parent variable used for grouping and aggregating
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated by the `parent_var` for 
            all numeric columns. Each observation of the parent variable will have 
            one row in the dataframe with the parent variable as the index. 
            The columns are also renamed using the `df_name`. Columns with all duplicate
            values are removed. 
    
    """
    
    # Remove id variables other than grouping variable
    for col in df:
        if col != parent_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    # Only want the numeric variables
    parent_ids = df[parent_var].copy()
    numeric_df = df.select_dtypes('number').copy()
    numeric_df[parent_var] = parent_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum'])

    # Need to create new column names
    columns = []

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        if var != parent_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))
    
    agg.columns = columns
    
    # Remove the columns with all redundant values
    _, idx = np.unique(agg, axis = 1, return_index=True)
    agg = agg.iloc[:, idx]
    
    return agg
def agg_categorical(df, parent_var, df_name):
    """
    Aggregates the categorical features in a child dataframe
    for each observation of the parent variable.
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    parent_var : string
        The variable by which to group and aggregate the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with aggregated statistics for each observation of the parent_var
        The columns are also renamed and columns with duplicate values are removed.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('category'))

    # Make sure to put the identifying id on the column
    categorical[parent_var] = df[parent_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(parent_var).agg(['sum', 'count', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['sum', 'count', 'mean']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    # Remove duplicate columns by values
    _, idx = np.unique(categorical, axis = 1, return_index = True)
    categorical = categorical.iloc[:, idx]
    
    return categorical
import gc

def agg_child(df, parent_var, df_name):
    """Aggregate a child dataframe for each observation of the parent."""
    
    # Numeric and then categorical
    df_agg = agg_numeric(df, parent_var, df_name)
    df_agg_cat = agg_categorical(df, parent_var, df_name)
    
    # Merge on the parent variable
    df_info = df_agg.merge(df_agg_cat, on = parent_var, how = 'outer')
    
    # Remove any columns with duplicate values
    _, idx = np.unique(df_info, axis = 1, return_index = True)
    df_info = df_info.iloc[:, idx]
    
    # memory management
    gc.enable()
    del df_agg, df_agg_cat
    gc.collect()
    
    return df_info
def agg_grandchild(df, parent_df, parent_var, grandparent_var, df_name):
    """
    Aggregate a grandchild dataframe at the grandparent level.
    
    Parameters
    --------
        df : dataframe
            Data with each row representing one observation
            
        parent_df : dataframe
            Parent table of df that must have the parent_var and 
            the grandparent_var. Used only to get the grandparent_var into
            the dataframe after aggregations
            
        parent_var : string
            Variable representing each unique observation in the parent.
            For example, `SK_ID_BUREAU` or `SK_ID_PREV`
            
        grandparent_var : string
            Variable representing each unique observation in the grandparent.
            For example, `SK_ID_CURR`. 
            
        df_name : string
            String for renaming the resulting columns.
            The columns are name with the `df_name` and with the 
            statistic calculated in the column
    
    Return
    --------
        df_info : dataframe
            A dataframe with one row for each observation of the grandparent variable.
            The grandparent variable forms the index, and the resulting dataframe
            can be merged with the grandparent to be used for training/testing. 
            Columns with all duplicate values are removed from the dataframe before returning.
    
    """
    
    # set the parent_var as the index of the parent_df for faster merges
    parent_df = parent_df[[parent_var, grandparent_var]].copy().set_index(parent_var)
    
    # Aggregate the numeric variables at the parent level
    df_agg = agg_numeric(df, parent_var, '%s_LOAN' % df_name)
    
    # Merge to get the grandparent variable in the data
    df_agg = df_agg.merge(parent_df, 
                          on = parent_var, how = 'left')
    
    # Aggregate the numeric variables at the grandparent level
    df_agg_client = agg_numeric(df_agg, grandparent_var, '%s_CLIENT' % df_name)
    
    # Can only apply one-hot encoding to categorical variables
    if any(df.dtypes == 'category'):
    
        # Aggregate the categorical variables at the parent level
        df_agg_cat = agg_categorical(df, parent_var, '%s_LOAN' % df_name)
        df_agg_cat = df_agg_cat.merge(parent_df,
                                      on = parent_var, how = 'left')

        # Aggregate the categorical variables at the grandparent level
        df_agg_cat_client = agg_numeric(df_agg_cat, grandparent_var, '%s_CLIENT' % df_name)
        df_info = df_agg_client.merge(df_agg_cat_client, on = grandparent_var, how = 'outer')
        
        gc.enable()
        del df_agg, df_agg_client, df_agg_cat, df_agg_cat_client
        gc.collect()
    
    # If there are no categorical variables, then we only need the numeric aggregations
    else:
        df_info = df_agg_client.copy()
    
        gc.enable()
        del df_agg, df_agg_client
        gc.collect()
    
    # Drop the columns with all duplicated values
    _, idx = np.unique(df_info, axis = 1, return_index=True)
    df_info = df_info.iloc[:, idx]
    
    return df_info
# Add domain features to base dataframe
app['LOAN_RATE'] = app['AMT_ANNUITY'] / app['AMT_CREDIT'] 
app['CREDIT_INCOME_RATIO'] = app['AMT_CREDIT'] / app['AMT_INCOME_TOTAL']
app['EMPLOYED_BIRTH_RATIO'] = app['DAYS_EMPLOYED'] / app['DAYS_BIRTH']
app['EXT_SOURCE_SUM'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis = 1)
app['EXT_SOURCE_MEAN'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
app['AMT_REQ_SUM'] = app[[x for x in app.columns if 'AMT_REQ_' in x]].sum(axis = 1)
bureau['LOAN_RATE'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM']
bureau_info = agg_child(bureau, 'SK_ID_CURR', 'BUREAU')
bureau_info.head()
bureau_info.shape
bureau_balance['PAST_DUE'] = bureau_balance['STATUS'].isin(['1', '2', '3', '4', '5'])
bureau_balance['ON_TIME'] = bureau_balance['STATUS'] == '0'
bureau_balance_info = agg_grandchild(bureau_balance, bureau, 'SK_ID_BUREAU', 'SK_ID_CURR', 'BB')
del bureau_balance, bureau
bureau_balance_info.head()
bureau_balance_info.shape
app = app.set_index('SK_ID_CURR')
app = app.merge(bureau_info, on = 'SK_ID_CURR', how = 'left')
del bureau_info
app.shape
app = app.merge(bureau_balance_info, on = 'SK_ID_CURR', how = 'left')
del bureau_balance_info
app.shape
previous = pd.read_csv('../input/previous_application.csv').replace({365243: np.nan})
previous = convert_types(previous)
previous['LOAN_RATE'] = previous['AMT_ANNUITY'] / previous['AMT_CREDIT']
previous["AMT_DIFFERENCE"] = previous['AMT_CREDIT'] - previous['AMT_APPLICATION']
previous_info = agg_child(previous, 'SK_ID_CURR', 'PREVIOUS')
previous_info.shape
app = app.merge(previous_info, on = 'SK_ID_CURR', how = 'left')
del previous_info
app.shape
installments = pd.read_csv('../input/installments_payments.csv').replace({365243: np.nan})
installments = convert_types(installments)
installments['LATE'] = installments['DAYS_ENTRY_PAYMENT'] > installments['DAYS_INSTALMENT']
installments['LOW_PAYMENT'] = installments['AMT_PAYMENT'] < installments['AMT_INSTALMENT']
installments_info = agg_grandchild(installments, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'IN')
del installments
installments_info.shape
app = app.merge(installments_info, on = 'SK_ID_CURR', how = 'left')
del installments_info
app.shape
cash = pd.read_csv('../input/POS_CASH_balance.csv').replace({365243: np.nan})
cash = convert_types(cash)
cash['LATE_PAYMENT'] = cash['SK_DPD'] > 0.0
cash['INSTALLMENTS_PAID'] = cash['CNT_INSTALMENT'] - cash['CNT_INSTALMENT_FUTURE']
cash_info = agg_grandchild(cash, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'CASH')
del cash
cash_info.shape
app = app.merge(cash_info, on = 'SK_ID_CURR', how = 'left')
del cash_info
app.shape
credit = pd.read_csv('../input/credit_card_balance.csv').replace({365243: np.nan})
credit = convert_types(credit)
credit['OVER_LIMIT'] = credit['AMT_BALANCE'] > credit['AMT_CREDIT_LIMIT_ACTUAL']
credit['BALANCE_CLEARED'] = credit['AMT_BALANCE'] == 0.0
credit['LOW_PAYMENT'] = credit['AMT_PAYMENT_CURRENT'] < credit['AMT_INST_MIN_REGULARITY']
credit['LATE'] = credit['SK_DPD'] > 0.0
credit_info = agg_grandchild(credit, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'CC')
del credit, previous
credit_info.shape
gc.collect()
gc.enable()
import time
time.sleep(600)
app = app.merge(credit_info, on = 'SK_ID_CURR', how = 'left')
del credit_info
app.shape
print('After manual feature engineering, there are {} features.'.format(app.shape[1] - 2))
gc.enable()
gc.collect()
print(f'Final size of data {return_size(app)}')
# Check for columns with duplicated values
# _, idx = np.unique(app, axis = 1, return_index = True)
# print('There are {} columns with all duplicated values.'.format(app.shape[1] - len(idx)))
app.to_csv('clean_manual_features.csv', chunksize = 100)
app.reset_index(inplace = True)
train, test = app[app['TARGET'].notnull()].copy(), app[app['TARGET'].isnull()].copy()
gc.enable()
del app
gc.collect()
import lightgbm as lgb

params = {'is_unbalance': True, 
              'n_estimators': 2673, 
              'num_leaves': 77, 
              'learning_rate': 0.00764, 
              'min_child_samples': 460, 
              'boosting_type': 'gbdt', 
              'subsample_for_bin': 240000, 
              'reg_lambda': 0.20, 
              'reg_alpha': 0.88, 
              'subsample': 0.95, 
              'colsample_bytree': 0.7}
train_labels = np.array(train.pop('TARGET')).reshape((-1, ))

test_ids = list(test.pop('SK_ID_CURR'))
test = test.drop(columns = ['TARGET'])
train = train.drop(columns = ['SK_ID_CURR'])

print('Training shape: ', train.shape)
print('Testing shape: ', test.shape)
model = lgb.LGBMClassifier(**params)
model.fit(train, train_labels)
preds = model.predict_proba(test)[:, 1]
submission = pd.DataFrame({'SK_ID_CURR': test_ids,
                           'TARGET': preds})

submission['SK_ID_CURR'] = submission['SK_ID_CURR'].astype(int)
submission['TARGET'] = submission['TARGET'].astype(float)
submission.to_csv('submission_manual.csv', index = False)
features = list(train.columns)
fi = pd.DataFrame({'feature': features,
                   'importance': model.feature_importances_})
def plot_feature_importances(df, n = 15, threshold = None):
    """
    Plots n most important features. Also plots the cumulative importance if
    threshold is specified and prints the number of features needed to reach threshold cumulative importance.
    Intended for use with any tree-based feature importances. 
    
    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be "feature" and "importance"
    
    n : int, default = 15
        Number of most important features to plot
    
    threshold : float, default = None
        Threshold for cumulative importance plot. If not provided, no plot is made
        
    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column
    
    Note
    --------
        * Normalization in this case means sums to 1. 
        * Cumulative importance is calculated by summing features from most to least important
    
    """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    
    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'blue', edgecolor = 'k', figsize = (12, 8),
                            legend = False)

    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
    plt.title(f'Top {n} Most Important Features', size = 18)
    plt.gca().invert_yaxis()
    
    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18);
        
        # Number of features needed for threshold cumulative importance
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        
        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.2, linestyles = '--', colors = 'red')
        plt.show();
        
        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 100 * threshold))
    
    return df
norm_fi = plot_feature_importances(fi, 25)
norm_fi.head(25)
