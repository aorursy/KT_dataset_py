# numpy and pandas for data manipulation

import numpy as np

import pandas as pd 



# sklearn preprocessing for dealing with categorical variables

from sklearn.preprocessing import LabelEncoder



# File system manangement

import os



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns
# data

app_train = pd.read_csv('../input/application_train.csv')

app_test = pd.read_csv('../input/application_test.csv')
# Create a label encoder object

le = LabelEncoder()

le_count = 0



# Iterate through the columns

for col in app_train:

    if app_train[col].dtype == 'object':

        # If 2 or fewer unique categories

        if len(list(app_train[col].unique())) <= 2:

            # Train on the training data

            le.fit(app_train[col])

            # Transform both training and testing data

            app_train[col] = le.transform(app_train[col])

            app_test[col] = le.transform(app_test[col])

            

            # Keep track of how many columns were label encoded

            le_count += 1
# one-hot encoding of categorical variables

app_train = pd.get_dummies(app_train)

app_test = pd.get_dummies(app_test)
train_labels = app_train['TARGET']



# Align the training and testing data, keep only columns present in both dataframes

app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)



# Add the target back in

app_train['TARGET'] = train_labels
app_train_domain = app_train.copy()

app_test_domain = app_test.copy()



app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']

app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']

app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']

app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']
app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']

app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']

app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']

app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']
from sklearn.preprocessing import MinMaxScaler, Imputer



# Drop the target from the training data

if 'TARGET' in app_train:

    train = app_train.drop(columns = ['TARGET'])

else:

    train = app_train.copy()

    

# Feature names

features = list(train.columns)



# Copy of the testing data

test = app_test.copy()



# Median imputation of missing values

imputer = Imputer(strategy = 'median')



# Scale each feature to 0-1

scaler = MinMaxScaler(feature_range = (0, 1))



# Fit on the training data

imputer.fit(train)



# Transform both training and testing data

train = imputer.transform(train)

test = imputer.transform(app_test)



# Repeat with the scaler

scaler.fit(train)

train = scaler.transform(train)

test = scaler.transform(test)



print('Training data shape: ', train.shape)

print('Testing data shape: ', test.shape)
from sklearn.linear_model import LogisticRegression



# Make the model with the specified regularization parameter

log_reg = LogisticRegression(C = 0.0001)



# Train on the training data

log_reg.fit(train, train_labels)
# Make predictions

# Make sure to select the second column only

log_reg_pred = log_reg.predict_proba(test)[:, 1]
# Submission dataframe

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = log_reg_pred



submit.to_csv('log_reg_baseline.csv', index = False)
from sklearn.ensemble import RandomForestClassifier



# Make the random forest classifier

random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
random_forest.fit(train, train_labels)

predictions = random_forest.predict_proba(test)[:, 1]
# Make a submission dataframe

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = predictions



# Save the submission dataframe

submit.to_csv('random_forest_baseline.csv', index = False)
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import gc



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

    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)

    

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

submission, fi, metrics = model(app_train, app_test)

print('Baseline metrics')

print(metrics)
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
fi_sorted = plot_feature_importances(fi)
submission.to_csv('baseline_lgb.csv', index = False)
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
from sklearn.model_selection import train_test_split

import xgboost as xgb



X_train,X_test,y_train,y_test = train_test_split(app_train,train_labels,test_size = 0.3,random_state = 1)

 

data_train = xgb.DMatrix(X_train, y_train)  # 使用XGBoost的原生版本需要对数据进行转化

data_test = xgb.DMatrix(X_test, y_test)

 

param = {'max_depth': 5, 'eta': 1, 'objective': 'binary:logistic'}

watchlist = [(data_test, 'test'), (data_train, 'train')]

n_round = 3

booster = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)

 

# 计算错误率

y_predicted = booster.predict(data_test)

y = data_test.get_label()

 

accuracy = sum(y == (y_predicted > 0.5))

accuracy_rate = float(accuracy) / len(y_predicted)

print ('样本总数：{0}'.format(len(y_predicted)))

print ('正确数目：{0}'.format(accuracy) )

print ('正确率：{0:.3f}'.format((accuracy_rate)))

datatest = xgb.DMatrix(app_test)

# 计算错误率

predictions = booster.predict(datatest)
# Make a submission dataframe

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = predictions



# Save the submission dataframe

submit.to_csv('xgb_baseline.csv', index = False)