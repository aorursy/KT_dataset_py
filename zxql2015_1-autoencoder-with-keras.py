# important packages to import

import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

import sys, os

import pickle

import gc; gc.enable()



from scipy import stats

from pylab import rcParams

from keras.models import Model, load_model

from keras.layers import Input, Dense

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import regularizers

from matplotlib import offsetbox

from matplotlib.ticker import NullFormatter

from sklearn import preprocessing, cross_validation, svm, manifold

from sklearn.cross_validation import cross_val_score, KFold

from sklearn.metrics import roc_curve, roc_auc_score, auc

from sklearn.ensemble import RandomForestClassifier # Load scikit's random forest classifier library

from sklearn.grid_search import GridSearchCV

from time import time

from datetime import datetime, timedelta

from collections import defaultdict

from multiprocessing import Pool, cpu_count



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Helper Functions



# holistic summary of the given data set. 

# "remove_bad_rowCol" can be turned on to remove non-informative col / row

def holistic_summary(df, remove_bad_rowCol = False, verbose = True):

    # remove non-informative columns

    if(remove_bad_rowCol):

        df = df.drop(df.columns[df.isnull().sum() >= .9 * len(df)], axis = 1)

        df = df.drop(df.index[df.isnull().sum(axis = 1) >= .5* len(df.columns)], axis = 0)

        

    # fix column names:

    df.columns = [c.replace(" ", "_").lower() for c in df.columns]

    

    print('***************************************************************')

    print('Begin holistic summary: ')

    print('***************************************************************\n')

    

    print('Dimension of df: ' + str(df.shape))

    print('Percentage of good observations: ' + str(1 - df.isnull().any(axis = 1).sum()/len(df)))

    print('---------------------------------------------------------------\n')

    

    print("Rows with nan values: " + str(df.isnull().any(axis = 1).sum()))

    print("Cols with nan values: " + str(df.isnull().any(axis = 0).sum()))

    print('Breakdown:')

    print(df.isnull().sum()[df.isnull().sum()!=0])

    print('---------------------------------------------------------------\n')

    

    print('Columns details: ')

    print('Columns with known dtypes: ')

    good_cols = pd.DataFrame(df.dtypes[df.dtypes!='object'], columns = ['type'])

    good_cols['nan_num'] = [df[col].isnull().sum() for col in good_cols.index]

    good_cols['unique_val'] = [df[col].nunique() for col in good_cols.index]

    good_cols['example'] = [df[col][1] for col in good_cols.index]

    good_cols = good_cols.reindex(good_cols['type'].astype(str).str.len().sort_values().index)

    print(good_cols)

    print('\n')

    

    try:

        print('Columns with unknown dtypes:')

        bad_cols = pd.DataFrame(df.dtypes[df.dtypes=='object'], columns = ['type'])

        bad_cols['nan_num'] = [df[col].isnull().sum() for col in bad_cols.index]

        bad_cols['unique_val'] = [df[col].nunique() for col in bad_cols.index]

        bad_cols['example(sliced)'] = [str(df[col][1])[:10] for col in bad_cols.index]

        bad_cols = bad_cols.reindex(bad_cols['example(sliced)'].str.len().sort_values().index)

        print(bad_cols)

    except Exception as e:

        print('No columns with unknown dtypes!')

    print('_______________________________________________________________\n\n\n')

    #if not verbose: enablePrint()

    return df



# fixing dtypes: time and numeric variables

def fix_dtypes(df, time_cols, num_cols):

    

    print('***************************************************************')

    print('Begin fixing data types: ')

    print('***************************************************************\n')

    

    def fix_time_col(df, time_cols):

        for time_col in time_cols:

            df[time_col] = pd.to_datetime(df[time_col], errors = 'coerce', format = '%Y%m%d')

        print('---------------------------------------------------------------')

        print('The following time columns has been fixed: ')

        print(time_cols)

        print('---------------------------------------------------------------\n')



    def fix_num_col(df, num_cols):

        for col in num_cols:

            df[col] = pd.to_numeric(df[col], errors = 'coerce')

        print('---------------------------------------------------------------')

        print('The following number columns has been fixed: ')

        print(num_cols)

        print('---------------------------------------------------------------\n')

        

    if(len(num_cols) > 0):

        fix_num_col(df, num_cols)

    fix_time_col(df, time_cols)



    print('---------------------------------------------------------------')

    print('Final data types:')

    result = pd.DataFrame(df.dtypes, columns = ['type'])

    result = result.reindex(result['type'].astype(str).str.len().sort_values().index)

    print(result)

    print('_______________________________________________________________\n\n\n')

    return df



# Load in user_logs

def transform_df(df):

    df = pd.DataFrame(df)

    df = df.sort_values(by=['date'], ascending=[False])

    df = df.reset_index(drop=True)

    df = df.drop_duplicates(subset=['msno'], keep='first')

    return df



def transform_df2(df):

    df = df.sort_values(by=['date'], ascending=[False])

    df = df.reset_index(drop=True)

    df = df.drop_duplicates(subset=['msno'], keep='first')

    return df



# Memory Reduction

def change_datatype(df):

    int_cols = list(df.select_dtypes(include=['int']).columns)

    for col in int_cols:

        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):

            df[col] = df[col].astype(np.int8)

        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):

            df[col] = df[col].astype(np.int16)

        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):

            df[col] = df[col].astype(np.int32)

        else:

            df[col] = df[col].astype(np.int64)

            

def plot_roc_curve(svm_clf, X_test, y_test, preds, isRF = False):

    from sklearn.metrics import roc_curve, roc_auc_score

    

    if isRF:

        y_score = svm_clf.predict_proba(X_test)[:,1]

    else:

        y_score = svm_clf.decision_function(X_test)

    (false_positive_rate, true_positive_rate, threshold) = roc_curve(y_test, y_score)

    roc_auc = auc(false_positive_rate, true_positive_rate)



    # Plot ROC curve

    plt.title('Receiver Operating Characteristic')

    plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], ls="--")

    # plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.legend(loc="lower right")

    plt.show()





# Print out the memory usage

def memo(df):

    mem = df.memory_usage(index=True).sum()

    print(mem/ 1024**2," MB")



# Print all the available files

def print_file():

    print(check_output(["ls", "../input"]).decode("utf8"))
print_file()
# Load in train and test

train = pd.read_csv('../input/train.csv')

train = train.append(pd.read_csv('../input/train_v2.csv'))

train.index = range(len(train))

test = pd.read_csv('../input/sample_submission_v2.csv')

# test = test.append(pd.read_csv('../input/sample_submission_zero.csv'))

# test.index = range(len(test))



# Load in other files

members = pd.read_csv('../input/members_v3.csv')

change_datatype(members)

print("Memo of members: ")

memo(members)



trans = pd.read_csv('../input/transactions.csv')

trans = trans.append(pd.read_csv('../input/transactions_v2.csv'))

trans.index = range(len(trans))

change_datatype(trans)

print("Memo of trans: ")

memo(trans)



# Loading in user_logs_v2.csv

df_iter = pd.read_csv('../input/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)

last_user_logs = []

i = 0 #~400 Million Records - starting at the end but remove locally if needed

for df in df_iter:

    if i>35: # used to be 35, just testing

        if len(df)>0:

            print(df.shape)

            p = Pool(cpu_count())

            df = p.map(transform_df, np.array_split(df, cpu_count()))   

            df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)

            df = transform_df2(df)

            p.close(); p.join()

            last_user_logs.append(df)

            print('...', df.shape)

            df = []

    i+=1



last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)

last_user_logs = transform_df2(last_user_logs)

# last_user_logs =  last_user_logs[['msno','num_100', 'num_25', 'num_unq', 'total_secs', 'date']]

print("Memo of last_user_logs: ")

memo(last_user_logs)
last_user_logs = last_user_logs.rename(columns = {'date':'last_user_log_date'})

last_user_logs.head()
# Only select 1/5% of train, merge with bigger csvs

# np.random.seed(47)

# samp = train  #.sample(frac = 1, replace = False)

train = train.merge(members, on = 'msno', how = 'left')

test = test.merge(members, on = 'msno', how = 'left')



temp_trans = trans.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)

temp_trans = temp_trans.drop_duplicates(subset=['msno'], keep='first')

temp_trans['discount'] = temp_trans['plan_list_price'] - temp_trans['actual_amount_paid']

temp_trans['amt_per_day'] = temp_trans['actual_amount_paid'] / temp_trans['payment_plan_days']

temp_trans['is_discount'] = temp_trans.discount.apply(lambda x: 1 if x > 0 else 0)

temp_trans['membership_days'] = pd.to_datetime(temp_trans['membership_expire_date']).subtract(pd.to_datetime(temp_trans['transaction_date'])).dt.days.astype(int)

train = train.merge(temp_trans, on = 'msno', how = 'left')

test = test.merge(temp_trans, on = 'msno', how = 'left')



temp_trans = []



train = train.merge(last_user_logs, on = 'msno', how = 'left')

test = test.merge(last_user_logs, on = 'msno', how = 'left')



last_user_logs = []
print("Train shape: ", train.shape)

# print("Samp shape: ", samp.shape)

print("Test shape: ", test.shape)



pd.set_option('max_columns', 100)

train.head()
test.head()
# samp = holistic_summary(samp)

train['last_user_log_date'] = train['last_user_log_date'].fillna(20170105.0)

train = fix_dtypes(train, time_cols = ['transaction_date', 'membership_expire_date', 'registration_init_time', 'last_user_log_date'], num_cols = [])
# test = holistic_summary(test)

test['last_user_log_date'] = test['last_user_log_date'].fillna(20170105.0)

test = fix_dtypes(test, time_cols = ['transaction_date', 'membership_expire_date', 'registration_init_time', 'last_user_log_date'], num_cols = [])
# 0. Date columns

print("Creating date columns... ")

# samp['last_user_log_date'] = samp['last_user_log_date'].fillna(np.mean(samp['last_user_log_date']))

date_dict = {'t_':'transaction_date', 'm_':'membership_expire_date', \

             'r_':'registration_init_time', 'l_':'last_user_log_date'}

for key in date_dict:  

    if key == 'r_':

        train[key+'month'] = [d.month for d in train[date_dict[key]]]

        train[key+'day'] = [d.day for d in train[date_dict[key]]]

#         samp[key+'wday'] = [d.weekday() for d in samp[date_dict[key]]]

    else:

        train[key+'day'] = [d.day for d in train[date_dict[key]]]

#         samp[key+'wday'] = [d.weekday() for d in samp[date_dict[key]]]

train['transaction_date'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in train['transaction_date']]

train['membership_expire_date'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in train['membership_expire_date']]

train['registration_init_time'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in train['registration_init_time']]

train['last_user_log_date'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in train['last_user_log_date']]

print("Done!")



print("Creating date columns for test... ")

for key in date_dict:  

    if key == 'r_':

        test[key+'month'] = [d.month for d in test[date_dict[key]]]

        test[key+'day'] = [d.day for d in test[date_dict[key]]]

#         test[key+'wday'] = [d.weekday() for d in test[date_dict[key]]]

    else:

        test[key+'day'] = [d.day for d in test[date_dict[key]]]

#         test[key+'wday'] = [d.weekday() for d in test[date_dict[key]]]

test['transaction_date'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in test['transaction_date']]

test['membership_expire_date'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in test['membership_expire_date']]

test['registration_init_time'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in test['registration_init_time']]

test['last_user_log_date'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in test['last_user_log_date']]

print("Done!")
# 1. number of transactions

print("Creating number of transactions... ")

ttemp = trans[['msno']]

temp = pd.DataFrame(ttemp['msno'].value_counts().reset_index())

temp.columns = ['msno','trans_count']

# train = pd.merge(train, transactions, how='left', on='msno')

train = pd.merge(train, temp, how='left', on='msno')

test = pd.merge(test, temp, how='left', on='msno')

temp = []; ttemp = []

print("Done!")
# 2. number of user logs in user_logs_v2 only

print("Creating number of user logs... ")

user_logs = pd.read_csv('../input/user_logs_v2.csv', usecols=['msno'])

user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())

user_logs.columns = ['msno','logs_count']

# train = pd.merge(train, user_logs, how='left', on='msno')

train = pd.merge(train, user_logs, how='left', on='msno')

test = pd.merge(test, user_logs, how='left', on='msno')

user_logs = []

print("Done!")
# 3. dummy encodings

print("Creating dummy encodings... ")

train['unique_index'] = range(len(train))

# Creat dummy variables for for following columns

prefix_dict = {'pm_id':'payment_method_id', 'pp_days':'payment_plan_days', 'city':'city',\

             'gender':'gender', 'reg_via':'registered_via', 'pl_price':'plan_list_price'}

dummy_dict = {'pm_id':[41,38], 'pp_days':[30], 'city':[22],

             'pl_price':[99, 149], 'reg_via':[7,4]} # ,22,38,39,35,29,36

for key in prefix_dict:

    if key in ['gender']:

        dummmm_df = pd.get_dummies(train[prefix_dict[key]])

        dummmm_df.columns = [key+'_'+str(s) for s in dummmm_df.columns]

        dummmm_df['unique_index'] = train['unique_index']

        train = train.merge(dummmm_df, on = 'unique_index', how = 'inner')

    else:

        for unique_val in dummy_dict[key]:

            train[key+'_'+str(unique_val)] = np.where(train[prefix_dict[key]] == unique_val, 1, 0)

#         samp[key+'_other'] = np.where(samp[prefix_dict[key]].isin(dummy_dict[key]), 0, 1)

    train = train.drop(prefix_dict[key], 1)

train = train.drop('unique_index', 1)

print("Done!")



print("Creating dummy encodings for test... ")

test['unique_index'] = range(len(test))

for key in prefix_dict:

    if key in ['gender']:

        dummmm_df = pd.get_dummies(test[prefix_dict[key]])

        dummmm_df.columns = [key+'_'+str(s) for s in dummmm_df.columns]

        dummmm_df['unique_index'] = test['unique_index']

        test = test.merge(dummmm_df, on = 'unique_index', how = 'inner')

    else:

        for unique_val in dummy_dict[key]:

            test[key+'_'+str(unique_val)] = np.where(test[prefix_dict[key]] == unique_val, 1, 0)

#         samp[key+'_other'] = np.where(samp[prefix_dict[key]].isin(dummy_dict[key]), 0, 1)

    test = test.drop(prefix_dict[key], 1)

test = test.drop('unique_index', 1)

print("Done!")

# 4. interaction terms

train['autorenew_&_not_cancel'] = ((train.is_auto_renew == 1) == (train.is_cancel == 0)).astype(np.int8)

test['autorenew_&_not_cancel'] = ((test.is_auto_renew == 1) == (test.is_cancel == 0)).astype(np.int8)



train['notAutorenew_&_cancel'] = ((train.is_auto_renew == 0) == (train.is_cancel == 1)).astype(np.int8)

test['notAutorenew_&_cancel'] = ((test.is_auto_renew == 0) == (test.is_cancel == 1)).astype(np.int8)



memo(train)

memo(test)
feature_cols = [col for col in train.columns if col not in ['is_churn', 'msno']]



train[feature_cols] = train[feature_cols].applymap(lambda x: np.nan if np.isinf(x) else x)

test[feature_cols] = test[feature_cols].applymap(lambda x: np.nan if np.isinf(x) else x)
print(train.columns[train.isnull().any()].tolist())

fill_dict = {}

for col in train.columns[train.isnull().any()].tolist():

    fill_dict[col] = np.mean(train[col])

train = train.fillna(value = fill_dict)

print(train.columns[train.isnull().any()].tolist())



print(test.columns[test.isnull().any()].tolist())

# fill_dict = {}

# for col in test.columns[test.isnull().any()].tolist():

#     fill_dict[col] = np.mean(test[col])

test = test.fillna(value = fill_dict)

print(test.columns[test.isnull().any()].tolist())
pd.set_option('max_columns', 100)

print(train.shape)

train.head()
print(test.shape)

test.head()
trans = []

members = []
print("Churn ratio", len(train[train['is_churn'] == 1])/len(train))
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,

                             roc_curve, recall_score, classification_report, f1_score,

                             precision_recall_fscore_support)
from math import log

def log_loss(preds, trues):

    preds = [max(min(s, 1-10**(-15)), 10**(-15)) for s in preds]

    return -np.mean([y * log(p) + (1-y) * log(1-p) for y,p in zip(trues, preds)])

trues  = [1,1,0,0]

preds  = [1,1,0,0]



print(log_loss(trues, preds))
feature_cols = [col for col in train.columns if col not in ['is_churn', 'msno']]

features = np.array(train[feature_cols])

response = np.array(train['is_churn'])

features_test = np.array(test[feature_cols])



print(features.shape)

print(response.shape)

print(features_test.shape)
from sklearn.preprocessing import StandardScaler

from sklearn import decomposition



# Create standardized features

features_std = StandardScaler().fit_transform(features)

features_std_test = StandardScaler().fit_transform(features_test)
from sklearn.metrics import roc_auc_score, roc_curve



X_train, X_test, y_train, y_test = cross_validation.train_test_split(features_std, response, test_size = 0.2)
# res = pd.DataFrame(res).sort_values('oob_score_', ascending = False)

leaf_size = 1

n_feautres = 10

# res.head(1)
def RF(X_train, X_test, y_train, y_test):

    clf = RandomForestClassifier(n_estimators = 250, n_jobs=-1, class_weight="balanced", \

                                 max_features = n_feautres, min_samples_leaf = leaf_size,\

                                 random_state = 47)

    print("Starting training...")

    clf.fit(X_train, y_train)

    print("Done")

    print("Prediction...")

    preds = clf.predict(X_test)

    # Print confusion matrix and plot ROC curve

    print(pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predicted']))

    plot_roc_curve(clf, X_test, y_test, preds, isRF = True)

    return clf



rf_clf = RF(X_train, X_test, y_train, y_test)
LABELS = ['not churn', 'churn']

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(features_std, response, test_size = 0.2)



y_pred = rf_clf.predict(X_test)

print(log_loss(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)



plt.figure(figsize=(10, 8))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()
# final_preds = rf_clf.predict(test[feature_cols])

# test['is_churn'] = final_preds.clip(0.+1e-15, 1-1e-15)

# test[['msno','is_churn']].to_csv('pred-rf.csv.gz', index=False, compression='gzip')
# Plot the feature importances of RF

importance = rf_clf.feature_importances_

importance = pd.DataFrame(importance, index=feature_cols, columns=["importance"])

importance["std"] = np.std([tree.feature_importances_ for tree in rf_clf.estimators_], axis=0)

importance = importance.sort_values('importance', ascending = False)

importance['col_names'] = importance.index



RF_important_cols = list(importance.index)[:20]

print("Most importance 15 features: ", RF_important_cols)

plt.figure(figsize=(10, 12))

sns.barplot(data = importance, y = 'col_names', x = 'importance')

# plt.xticks(rotation=90)
# from sklearn.svm import LinearSVC



# def SVC(features, response):

#     X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, response, test_size = 0.2)



#     clf = LinearSVC(class_weight="balanced")

#     clf.fit(X_train, y_train)

#     preds = clf.predict(X_test)

#     # Print confusion matrix and plot ROC curve

#     print(pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predicted']))

#     plot_roc_curve(clf, X_test, y_test, preds)



# SVC(features, response)

# SVC(features_std, response)

# # SVC(features_pca, response)
np.random.seed(47)

train = train.sample(frac = 0.2, replace = False)
from sklearn import decomposition

from sklearn.preprocessing import StandardScaler



# Redefine the featuresets for Autoencoder

features = np.array(train[RF_important_cols])

features_test = np.array(test[RF_important_cols])

response = np.array(train['is_churn'])



# Create standardized features

features_std = StandardScaler().fit_transform(features)

features_std_test = StandardScaler().fit_transform(features_test)



# Create a pca object with the 10 components

pca = decomposition.PCA(n_components=10)



# Fit the PCA and transform the data

features_pca = pca.fit_transform(features_std)

features_pca_test = pca.fit_transform(features_std)
# Convert features_pca back to a df and add the is_churn column

features_std = pd.DataFrame(features_std)

features_std['is_churn'] = np.array(train['is_churn'])



features_std_test = pd.DataFrame(features_std).values





# features_std = pd.DataFrame(features_pca)

# features_std['is_churn'] = np.array(train['is_churn'])



# features_std_test = pd.DataFrame(features_pca_test).values
from sklearn.cross_validation import train_test_split

X_train, X_test = train_test_split(features_std, test_size=0.2, random_state=47)

X_train = X_train[X_train['is_churn'] == 0]

X_train = X_train.drop(['is_churn'], axis=1)



y_test = X_test['is_churn']

X_test = X_test.drop(['is_churn'], axis=1)



X_train = X_train.values

X_test = X_test.values
print(X_train.shape)

print(X_test.shape)
input_dim = X_train.shape[1]

encoding_dim = 8
input_layer = Input(shape=(input_dim, ))



encoder = Dense(encoding_dim, activation="tanh", 

                activity_regularizer=regularizers.l1(10e-5))(input_layer)

encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)



decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)

decoder = Dense(input_dim, activation='relu')(decoder)



autoencoder = Model(inputs=input_layer, outputs=decoder)
nb_epoch = 100

batch_size = 32



autoencoder.compile(optimizer='adam', 

                    loss='mean_squared_error', 

                    metrics=['accuracy'])



checkpointer = ModelCheckpoint(filepath="model.h5",

                               verbose=0,

                               save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs',

                          histogram_freq=0,

                          write_graph=True,

                          write_images=True)
history = autoencoder.fit(X_train, X_train,

                    epochs=nb_epoch,

                    batch_size=batch_size,

                    shuffle=True,

                    validation_data=(X_test, X_test),

                    verbose=1,

                    callbacks=[checkpointer, tensorboard]).history
print(checkpointer)

print(tensorboard)

print(nb_epoch)

print(batch_size)

print(X_train)

autoencoder = load_model('model.h5')
plt.plot(history['loss'])

plt.plot(history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right');
predictions = autoencoder.predict(X_test)

mse = np.mean(np.power(X_test - predictions, 2), axis=1)

error_df = pd.DataFrame({'reconstruction_error': mse,

                        'true_class': y_test})
error_df.describe()
fig = plt.figure()

ax = fig.add_subplot(111)

normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]

_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10, normed = True)

plt.title('Reconstruction error without churn group')

sns.despine()
fig = plt.figure()

ax = fig.add_subplot(111)

fraud_error_df = error_df[error_df['true_class'] == 1]

_ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10, normed = True)

plt.title('Reconstruction error with churn group')

sns.despine()
fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'g--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show();
precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)

plt.plot(recall, precision, 'b', label='Precision-Recall curve')

plt.title('Recall vs Precision')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.show()
plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')

plt.title('Precision for different threshold values')

plt.xlabel('Threshold')

plt.ylabel('Precision')

plt.show()
plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')

plt.title('Recall for different threshold values')

plt.xlabel('Reconstruction error')

plt.ylabel('Recall')

plt.show()
threshold = 1
groups = error_df.groupby('true_class')

fig, ax = plt.subplots()



for name, group in groups:

    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',

            label= "churn" if name == 1 else "not churn", alpha = 0.5)

ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')

ax.legend()

plt.title("Reconstruction error for different classes")

plt.ylabel("Reconstruction error")

plt.xlabel("Data point index")

plt.ylim(0,8)

sns.despine()

plt.show();
LABELS = ['not churn', 'churn']



y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

print(log_loss(error_df.true_class, y_pred))



conf_matrix = confusion_matrix(error_df.true_class, y_pred)

plt.figure(figsize=(10,8))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()
predictions = autoencoder.predict(features_std_test)

mse = np.mean(np.power(features_std_test - predictions, 2), axis=1)

error_df = pd.DataFrame({'reconstruction_error': mse})



y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]



test['is_churn'] = y_pred.clip(0.+1e-15, 1-1e-15)

test[['msno','is_churn']].to_csv('pred-auto.csv.gz', index=False, compression='gzip')