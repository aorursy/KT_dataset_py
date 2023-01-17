# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, Imputer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train_control = pd.read_csv('../input/application_train.csv')
test_control = pd.read_csv('../input/application_test.csv')

training_dataset = pd.read_csv('../input/application_train.csv')
test_dataset = pd.read_csv('../input/application_test.csv')

previous_loan_information = pd.read_csv('../input/POS_CASH_balance.csv')
previous_application_data = pd.read_csv('../input/bureau.csv')

bureau_balance = pd.read_csv('../input/bureau_balance.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
previous_application = pd.read_csv('../input/previous_application.csv')

# Any results you write to the current directory are saved as output.
previous_application.head()

# Group by the client id, calculate aggregation statistics
###previous_application_data_agg = previous_application_data.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
###previous_application_data_agg.head()
#### List of column names
###columns = ['SK_ID_CURR']
###
#### Iterate through the variables names
###for var in previous_application_data_agg.columns.levels[0]:
###    # Skip the id name
###    if var != 'SK_ID_CURR':
###        
###        # Iterate through the stat names
###        for stat in previous_application_data_agg.columns.levels[1][:-1]:
###            # Make a new column name for the variable and stat
###            columns.append('bureau_%s_%s' % (var, stat))
#### Assign the list of columns names as the dataframe column names
###previous_application_data_agg.columns = columns
###previous_application_data_agg.head()
##previous_application_data['CREDIT_DAY_OVERDUE'].value_counts()

number_active_loans = previous_application_data[previous_application_data['CREDIT_ACTIVE'] == 'Active'].groupby(['SK_ID_CURR'], as_index = False).agg({'CREDIT_CURRENCY': "count", 'CREDIT_DAY_OVERDUE':"sum"})
number_active_loans.columns = ['SK_ID_CURR', 'num_active_loans', 'num_days_overdue']
number_active_loans
training_dataset = training_dataset.merge(number_active_loans, left_on = 'SK_ID_CURR', right_on = 'SK_ID_CURR')
test_dataset = test_dataset.merge(number_active_loans, left_on = 'SK_ID_CURR', right_on = 'SK_ID_CURR')
## what does the target variable look like?
training_dataset['TARGET'].value_counts()
# Number of unique classes in each object column
training_dataset.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
### we need to convert the categorial variables to numeric values.. one hot encoding..
# one-hot encoding of categorical variables
training_dataset = pd.get_dummies(training_dataset)
test_dataset = pd.get_dummies(test_dataset)

## one hot encoding significantly increases the number of variables.. it would help to try dimensionality reduction to reduce the number of variables

#### since we one hot encoded, now we need to align the two datasets 
train_labels = training_dataset['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
training_dataset, test_dataset = training_dataset.align(test_dataset, join = 'inner', axis = 1)

# Add the target back in
training_dataset['TARGET'] = train_labels

### checking if the two datasets have the same columns
print('Training Features shape: ', training_dataset.shape)
print('Testing Features shape: ', test_dataset.shape)
#### The next thing to check is to check if there are large outliers -- we can do this with the describe function

training_dataset['DAYS_EMPLOYED'].describe()
### so days employed seems to have at least one outlier.. next question is what to do with this outlier.. 
#### first step is to check if the outliers have a lower or higher rate of default compared to the rest of the data

anom = training_dataset[training_dataset['DAYS_EMPLOYED'] == 365243]
non_anom = training_dataset[training_dataset['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))
#### a good solution here is to replace the outlier with a missing value (np.nan) and then create a flag indicating that the data had a missing value originally

# Create an anomalous flag column
training_dataset['DAYS_EMPLOYED_ANOM'] = training_dataset["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
training_dataset['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

### do the same for the test dataset
test_dataset['DAYS_EMPLOYED_ANOM'] = test_dataset["DAYS_EMPLOYED"] == 365243
test_dataset["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
#### Next thing to check is correlation with independent variables and the target variable
# Find correlations with the target and sort
correlations = training_dataset.corr()['TARGET'].sort_values()
# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

### checking the predictability of a few variables

plt.figure(figsize = (10, 12))

# iterate through the sources
for i, source in enumerate(['num_active_loans', 'num_days_overdue']):
    
    # create a new subplot for each source
    plt.subplot(3, 1, i + 1)
    # plot repaid loans
    sns.kdeplot(training_dataset.loc[training_dataset['TARGET'] == 0, source], label = 'target == 0')
    # plot loans that were not repaid
    sns.kdeplot(training_dataset.loc[training_dataset['TARGET'] == 1, source], label = 'target == 1')
    
    # Label the plots
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)
def agg_numeric(df, group_var, df_name):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        df (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg


def count_categorical(df, group_var, df_name):
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    return categorical


# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
# Plots the disribution of a variable colored by value of the target
def kde_target(var_name, df):

    df = df[np.isfinite(df[var_name])] #drop nans

    # Calculate the correlation coefficient between the new variable and the target
    corr = df['TARGET'].corr(df[var_name])

    # Calculate medians for repaid vs not repaid
    avg_repaid = df.ix[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.ix[df['TARGET'] == 1, var_name].median()

    plt.figure(figsize = (12, 6))

    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.ix[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.ix[df['TARGET'] == 1, var_name], label = 'TARGET == 1')

    # label the plot
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend();

    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)
    

# Function to calculate correlations with the target for a dataframe
def target_corrs(df):

    # List of correlations
    corrs = []

    # Iterate through the columns 
    for col in df.columns:
        print(col)
        # Skip the target column
        if col != 'TARGET':
            # Calculate correlation with the target
            corr = df['TARGET'].corr(df[col])

            # Append the list as a tuple
            corrs.append((col, corr))
            
    # Sort by absolute magnitude of correlations
    corrs = sorted(corrs, key = lambda x: abs(x[1]), reverse = True)
    
    return corrs


previous_application_counts = count_categorical(previous_application, group_var = 'SK_ID_CURR', df_name = 'previous_application')
previous_application_agg_new = agg_numeric(previous_application_data.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'previous_application_data')

bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')


### This merges all of the calculated metrics by loan
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')
bureau_by_loan = bureau_by_loan.merge(previous_application_data[['SK_ID_BUREAU', 'SK_ID_CURR']], on = 'SK_ID_BUREAU', how = 'left')

### This takes the calculated metrics by loan and aggregates them on client id
bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')

bureau_balance_counts.head()
previous_application_agg_new.head()




training_dataset.head()
####### Do a little feature engineering
training_dataset['ANNUITY_INCOME_PERCENT'] = training_dataset['AMT_ANNUITY'] / training_dataset['AMT_INCOME_TOTAL']
test_dataset['ANNUITY_INCOME_PERCENT'] = test_dataset['AMT_ANNUITY'] / test_dataset['AMT_INCOME_TOTAL']
# Merge with the value counts of bureau
training_dataset = training_dataset.merge(previous_application_counts, on = 'SK_ID_CURR', how = 'left')

# Merge with the stats of bureau
training_dataset = training_dataset.merge(previous_application_agg_new, on = 'SK_ID_CURR', how = 'left')

# Merge with the monthly information grouped by client
training_dataset = training_dataset.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
#### There are a bunch of missing values in our new variables.. we should look into these
missing_train = missing_values_table(training_dataset)
missing_train.head(10)
# Merge with the value counts of bureau
test_dataset = test_dataset.merge(previous_application_counts, on = 'SK_ID_CURR', how = 'left')

# Merge with the stats of bureau
test_dataset = test_dataset.merge(previous_application_agg_new, on = 'SK_ID_CURR', how = 'left')

# Merge with the value counts of bureau balance
test_dataset = test_dataset.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
train_labels = training_dataset['TARGET']

# Align the dataframes, this will remove the 'TARGET' column
training_dataset, test_dataset = training_dataset.align(test_dataset, join = 'inner', axis = 1)

training_dataset['TARGET'] = train_labels
##### Drop variables with x% of observations that are missing
missing_test = missing_values_table(test_dataset)
missing_test_vars = list(missing_test.index[missing_test['% of Total Values'] > 90])

missing_train = missing_values_table(training_dataset)
missing_train_vars = list(missing_test.index[missing_test['% of Total Values'] > 90])

missing_columns = list(set(missing_test_vars + missing_train_vars))

# Drop the missing columns
training_dataset = training_dataset.drop(columns = missing_columns)
test_dataset = test_dataset.drop(columns = missing_columns)

print('There are %d columns with more than 90%% missing in either the training or testing data.' % len(missing_columns))
########## Save the training and test datasets
training_dataset.to_csv('train_bureau_raw.csv', index = False)
test_dataset.to_csv('test_bureau_raw.csv', index = False)
kde_target(var_name='previous_application_data_CREDIT_ACTIVE_Active_count_norm', df=training_dataset)
training_dataset.head()
for x in training_dataset.columns:
    if "CREDIT_ACTIVE" in x:
        print(x)
    else:
        pass






# Drop the target from the training data
if 'TARGET' in training_dataset:
    train = training_dataset.drop(columns = ['TARGET'])
else:
    train = training_dataset.copy()
    
# Feature names
features = list(train.columns)

# Copy of the testing data
test = test_dataset.copy()

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(test_dataset)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)



##from sklearn.ensemble import RandomForestClassifier
##
### Make the random forest classifier
##random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
### Train on the training data
##random_forest.fit(train, train_labels)
##
### Extract feature importances
##feature_importance_values = random_forest.feature_importances_
##feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
##
### Make predictions on the test data
##predictions = random_forest.predict_proba(test)[:, 1]
### Make a submission dataframe
##submit = test_dataset[['SK_ID_CURR']]
##submit['TARGET'] = predictions
##
### Save the submission dataframe
##submit.to_csv('random_forest_baseline.csv', index = False)

import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import gc

import matplotlib.pyplot as plt
def model(features, test_features, encoding = 'ohe', n_folds = 5):
    
    """Train and test a light gradient boosting model using
    cross validation. 
    
    Parameters
    --------
        features (pd.DataFrame): 
            dataframe of training features to use 
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame): 
            dataframe of testing features to use
            for making predictions with the model. 
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): 
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame): 
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame): 
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
        
    """
    
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        # Create a label encoder
        label_encoder = LabelEncoder()
        
        # List for storing categorical indices
        cat_indices = []
        
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # Extract feature names
    feature_names = list(features.columns)
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = False, random_state = 50)
    
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics
def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest) 
        with a column for normalized importance
        """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df
submission, fi, metrics = model(train_control, test_control)
fi_sorted = plot_feature_importances(fi)
submission.to_csv('control.csv', index = False)






