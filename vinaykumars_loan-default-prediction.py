# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('/kaggle/input/loan-default-prediction/train_v2.csv.zip')
#checking the first 5 rows of training data

data_train.head()
#shape of training data

data_train.shape
#information about data and it's types

data_train.info()
#selecting categorical columns

data_cat = data_train.select_dtypes(include=['object']).copy()
data_cat.head(10)
print(data_cat.columns)

print(data_cat['f137'].value_counts())
#extracting numerical columns from training data

data_num = data_train.select_dtypes(include=['float64', 'int64']).copy()
data_num.head()
data_num.columns
#checking for any missing values in each column

data_num.isnull().sum()
#imputing missing values with mean of that column

data_num_imputed = data_num.fillna(data_num.mean(), inplace=False)
#checking for missing values

data_num_imputed.isnull().sum()
#removing id column from imputed data

data_num_imputed.drop(columns='id', inplace=True)
data_num_imputed.head()
#shuffling dataframe in random order to remove any bias due to time sampling

data_num_shuffled = data_num_imputed.sample(frac=1, random_state=2)

data_num_shuffled.head()
#seperating loss column and calculating correlation matrix for features

loss = data_num_shuffled['loss'].copy()

data_num_shuffled.drop(columns='loss', inplace=True)

corr_matrix = data_num_shuffled.corr().abs()

corr_matrix.head()

corr_matrix.shape
#taking upper triangular part of correlation matrix 

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper.head()
# extracting highly correlated features with correlation coefficient of more than 0.8

threshold = 0.8

col_to_drop = [column for column in upper.columns if any(upper[column]>threshold)]

len(col_to_drop)
# dropping highly correlated columns

data_num_shuffled.drop(columns=col_to_drop, inplace=True)

data_num_shuffled.columns
# finding correlation of features with target values of loss and convert into a dataframe

corr_tar = data_num_shuffled.corrwith(loss).sort_values()

print(corr_tar.head(30))

print(corr_tar.tail(30))

corr_tar_df = corr_tar.to_frame().transpose()

corr_tar_df.isna()
# extracting features having NaN value correlation with loss to remove them 

col_to_drop_1 = corr_tar_df.columns[corr_tar_df.isna().any()].to_list()

print(len(col_to_drop_1))

print(col_to_drop_1)
data_num_shuffled.drop(columns=col_to_drop_1, inplace=True)
data_num_shuffled.columns
# creating a copy of final features dataframe 

X_feat = data_num_shuffled.copy()
#creating a copy of loss dataframe 

Y_tar = loss.copy()
# making target or loss a binary valued variable for better classification and interpretation

Y_tar[Y_tar>0] = 1

Y_tar.value_counts()
# splitting data into train and test splits and normalizing data for better training

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_feat, Y_tar, test_size=0.2, random_state=42)

ss = StandardScaler()

X_train = ss.fit_transform(x_train)

X_test = ss.transform(x_test)
# converting y to one-dimensional array

y_train = np.array(y_train).reshape((-1, ))

y_test = np.array(y_test).reshape((-1, ))
# function for calculating cross validation scores for evaluating model performance on train data and test data

def scores(X_train, y_train, X_test, y_test, model):

    from sklearn.model_selection import cross_val_score, ShuffleSplit

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42) # shufflesplit is a cross validation strategy that is best used when n_features < n_samples

    train_scores = cross_val_score(estimator = model, X = X_train, y = y_train, cv = cv)

    test_scores = cross_val_score(estimator = model, X = X_test, y = y_test, cv = cv)

    return [round(train_scores.mean(), 4), round(test_scores.mean(),4)]
# function to fit and evaluate model performance

def fit_and_evaluate(model):

    

    # Train the model

    model.fit(X_train, y_train)

    

    # Make predictions and evalute

    y_pred = model.predict(X_test)

    cross_val_scores = scores(X_train, y_train, X_test, y_test, model)

    

    # calculate accuracy and f_score for model performance

    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(y_test, y_pred, normalize=True)

    f_score = f1_score(y_test, y_pred, average='macro')

    

    # dictionary for storing all the above values

    metric = dict()

    metric['cv_scores'] = cross_val_scores

    metric['accuracy'] = round(accuracy,4)

    metric['f_score'] = round(f_score,4)

    

    # Return the performance metric 

    return metric
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



# logistic regression 

logistic = LogisticRegression(max_iter = 50, random_state=42)

log_c = fit_and_evaluate(logistic)

print('performance of logistic regression on train and test data:', log_c['cv_scores'])

log_c.pop('cv_scores')

print(log_c)

df_log = pd.DataFrame(log_c, index=[0])

print(df_log)



# Random forest classification

RFC = RandomForestClassifier(n_estimators = 15, random_state=42)

random = fit_and_evaluate(RFC)

print('performance of Random Forest Classifier on train and test data:', random['cv_scores'])

random.pop('cv_scores')

print(random)

df_rand = pd.DataFrame(random, index=[1])

print(df_rand)



# Xgboost classification

gb = XGBClassifier()

gb_c = fit_and_evaluate(gb)

print('performance of XGB Classifier on train and test data:', gb_c['cv_scores'])

gb_c.pop('cv_scores')

print(gb_c)

df_gb = pd.DataFrame(gb_c, index=[2])

print(df_gb)



frames = [df_log, df_rand, df_gb]

model_compare_df = pd.concat(frames)



print(model_compare_df)
# finding least important features from random forest classification so as to see if we can improve our model performance or we can get same performance with less features

feature_importances = RFC.feature_importances_

feature_importances = pd.DataFrame({'feature': list(X_feat.columns), 'importance': feature_importances}).sort_values('importance', ascending = True)

print(feature_importances['importance'].mean()/2) # threshold for filtering least important features

least_important_feat = list(feature_importances[feature_importances['importance'] < 0.0025]['feature'])

len(least_important_feat)
# creating dataframe with important features

X_feat_imp = X_feat.drop(columns=least_important_feat, inplace=False)

len(X_feat_imp.columns)
# splitting data into train and test

x_train_imp, x_test_imp, y_train, y_test = train_test_split(X_feat_imp, Y_tar, test_size=0.2, random_state=42)

ss = StandardScaler()

X_train_imp = ss.fit_transform(x_train_imp)

X_test_imp = ss.transform(x_test_imp)
# converting y to one-dimensional array

y_train = np.array(y_train).reshape((-1, ))

y_test = np.array(y_test).reshape((-1, ))
# function to fit and evaluate model performance with important features

def fit_and_evaluate_imp(model):

    

    # Train the model

    model.fit(X_train_imp, y_train)

    

    # Make predictions and evalute

    y_pred = model.predict(X_test_imp)

    cross_val_scores = scores(X_train_imp, y_train, X_test_imp, y_test, model)

    

    # calculate accuracy and f_score for model performance

    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(y_test, y_pred, normalize=True)

    f_score = f1_score(y_test, y_pred, average='macro')

    

    # dictionary for storing all the above values

    metric = dict()

    metric['cv_scores'] = cross_val_scores

    metric['accuracy'] = round(accuracy,4)

    metric['f_score'] = round(f_score,4)

    

    # Return the performance metric 

    return metric
# Random forest classification with important features

RFC_imp = RandomForestClassifier(n_estimators = 15, random_state=42)

random_imp = fit_and_evaluate_imp(RFC_imp)

print('performance of Random Forest Classifier on train and test data:', random_imp['cv_scores'])

random_imp.pop('cv_scores')

print(random_imp)

df_rand_imp = pd.DataFrame(random_imp, index=[0])

print(df_rand_imp)
data_test = pd.read_csv('/kaggle/input/loan-default-prediction/test_v2.csv.zip')
data_test.shape
data_test.columns
print(X_feat.columns)

print(X_feat_imp.columns)
# getting columns that are used for training

test_feat = data_test[X_feat.columns]

test_feat_imp = data_test[X_feat_imp.columns]

print(test_feat.columns)

print(test_feat_imp.columns)
print(test_feat.isnull().sum())

print(test_feat_imp.isnull().sum())
test_feat_imputed = test_feat.fillna(test_feat.mean(),inplace=False)

test_feat_imp_imputed = test_feat_imp.fillna(test_feat_imp.mean(),inplace=False)
print(test_feat_imputed.isnull().sum())

print(test_feat_imp_imputed.isnull().sum())
ss = StandardScaler()

test_feat_scaled = ss.fit_transform(test_feat_imputed)

test_feat_imp_scaled = ss.fit_transform(test_feat_imp_imputed)
predicted_val = RFC.predict(test_feat_scaled)

predicted_val_imp = RFC_imp.predict(test_feat_imp_scaled)

df_val = pd.DataFrame({'loss_as_binary':predicted_val})

df_val_imp = pd.DataFrame({'loss_as_binary':predicted_val_imp})

print(df_val)

print(df_val_imp)
predicted_prob = RFC.predict_proba(test_feat_scaled)

predicted_prob_imp = RFC_imp.predict_proba(test_feat_imp_scaled)

df_prob = pd.DataFrame({'no_default': predicted_prob[:, 0], 'default': predicted_prob[:, 1]})

df_prob_imp = pd.DataFrame({'no_default': predicted_prob_imp[:, 0], 'default': predicted_prob_imp[:, 1]})

print(df_prob)

print(df_prob_imp)
submission = pd.DataFrame()

submission['id'] = data_test.id

submission['loss_as_binary'] = df_val.loss_as_binary

submission['default_prob'] = round(df_prob.default,2)

print(submission)
submission.head()
Submission = submission.to_csv(index=False)
submission_imp = pd.DataFrame()

submission_imp['id'] = data_test.id

submission_imp['loss_as_binary'] = df_val_imp.loss_as_binary

submission_imp['default_prob'] = round(df_prob_imp.default,2)

print(submission_imp)
submission_imp.head()
Submission_imp = submission_imp.to_csv(index=False)
import os 

os.chdir(r'/kaggle/working')
submission.to_csv(r'SUBMISSION.csv',index=False)

submission_imp.to_csv(r'SUBMISSION_imp.csv',index=False)
from IPython.display import FileLink

FileLink(r'SUBMISSION.csv')
FileLink(r'SUBMISSION_imp.csv')