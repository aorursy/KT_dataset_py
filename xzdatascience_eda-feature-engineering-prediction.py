import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import datetime

import gc

import warnings

warnings.filterwarnings("ignore")
from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.preprocessing import minmax_scale

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from hyperopt import hp, tpe

from hyperopt.fmin import fmin
#!pip install xgboost
# read in files

raw_train_identity  = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")

raw_train_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")

raw_test_identity  = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")

raw_test_transaction = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")
# reduce file memory of train_identity, train_transaction, test_identity, and test_transaction 

# credit to @Tharindu Gangoda: https://www.kaggle.com/tharug/ieee-fraud-detection

def memory_reduction(df):

    # check original dataframe usage

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    # reduce all numeric columns to a smaller data type

    for col in df.columns:

        col_type = df[col].dtype.name

        if col_type not in ('object', 'category'):

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



        else:

            pass

    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
# reduce original data memory

raw_test_identity = memory_reduction(raw_test_identity)

raw_test_transaction = memory_reduction(raw_test_transaction)

raw_train_transaction = memory_reduction(raw_train_transaction)

raw_train_identity = memory_reduction(raw_train_identity)
# preview raw_train_transaction 

raw_train_transaction.head()
# create a bar graph to visualize the distribution of fraud vs. non fraud transactions

plt.figure(figsize=(12,6))

plt.title('Fraud Transaction Distribution')

ax=sns.countplot(x='isFraud', data =raw_train_transaction)

plt.ylabel("Transaction Count")

for p in ax.patches:

             ax.annotate(str(round(p.get_height()/len(raw_train_transaction)*100,2))+"%", (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),

                 textcoords='offset points')

# we examine whether the continous variable TransactionAMT has normal distribution or not by plotting a histogram 

plt.figure(figsize=(12,6))

sns.distplot(raw_train_transaction['TransactionAmt'])

plt.title("TransactionAmt Distribution ")

plt.ylabel("Probability Density")

plt.show()
plt.figure(figsize=(12,6))

sns.distplot(raw_train_transaction['TransactionAmt'].apply(np.log))

plt.title("Log Transformation of TransactionAmt Distribution ")

plt.ylabel("Probability Density")

plt.show()
# defind a function to visualize the percentage of fraud transactions in each categorical variables 

def find_percentage(df,variable_name,fig_size,rotate):

    # create empty list to sore percentage 

    find_percent=[]

    # go through all the columns 

    for i in variable_name:

        # create a dictionary

        percent=dict()

        # fill na with 'NA'

        df[i] = df[i].fillna("NA")

        # go through each category

        for j in sorted(df[i].unique()):

            # fraud number

            fraud_num = len(df[(df[i]==j) & (df['isFraud']==1)])

            # not fraud number

            total_num = len(df[df[i]==j])

            # round to 2 decimal

            temp = dict([(j,round((fraud_num/total_num*100),2))])

            # add new key &value

            percent.update(temp)

        plt.figure(figsize=fig_size)

        plt.title("Number of Transactions for " + str(i)) 

        ax=sns.countplot(x=i,data=df,order=sorted(df[i].unique()))

        plt.ylabel("Transaction Count")

        # add value label

        for p in ax.patches:

            ax.annotate(str(round(p.get_height()/len(raw_train_transaction)*100,2))+"%", (p.get_x() + p.get_width() / 2., p.get_height()),

            ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')

        plt.xticks(rotation=rotate)

        plt.figure(figsize=fig_size)

        plt.title("Fraud Transaction Percentage for " + str(i))

        ax=sns.barplot(x=list(percent.keys()),

            y=list(percent.values()))        

        plt.xticks(range(len(percent)), list(percent.keys()))

        plt.xticks(rotation=rotate)

        plt.xlabel(i)

        plt.ylabel("Fraud Transaction Percentage of Each Category")

        # add value label

        for p in ax.patches:

            ax.annotate(str(p.get_height())+'%', (p.get_x() + p.get_width() / 2., p.get_height()),

            ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),

            textcoords='offset points')

        find_percent.append(percent)

# find out the number of transactions for each category of ProductCD variable

# find out the fraud transaction percentage of total transactions for each category of ProductCD variable

fig_size =(12,6)

rotation=0

find_percentage(raw_train_transaction,['ProductCD'],fig_size,rotation)
# check the distribution for continuous card1 variable 

plt.figure(figsize=(12,6))

sns.distplot(raw_train_transaction[raw_train_transaction['isFraud']==0]['card1'])

sns.distplot(raw_train_transaction[raw_train_transaction['isFraud']==1]['card1'])

plt.legend(labels=['Not Fraud','Fraud'])

plt.show()
# find out the number of transactions for each category of card4 variable

# find out the fraud transaction percentage of total transactions for each category of card4 variable

find_percentage(raw_train_transaction,['card4'],fig_size,rotation)
# find out the number of transactions for each category of card6 variable

# find out the fraud transaction percentage of total transactions for each category of card6 variable

find_percentage(raw_train_transaction,['card6'],fig_size,rotation)
# create list to store M1-M9 variable names

M_cols = ["M"+str(i) for i in np.arange(1, 10, 1)]
# find out the number of transactions for each category of M1 to M9 variables

# find out the fraud transaction percentage of total transactions for each category of M1 to M9 variables

find_percentage(raw_train_transaction,M_cols,fig_size,rotation)
# visualize the distribution of each addr1 category

plt.figure(figsize=(12,6))

raw_train_transaction['addr1'].plot(kind='hist',bins=80)

plt.xticks(np.arange(min(raw_train_transaction['addr1']), max(raw_train_transaction['addr1'])+1, 20))

plt.show()
# visualize the distribution of each addr2 category

plt.figure(figsize=(12,6))

raw_train_transaction['addr2'].plot(kind='hist',bins=80)

plt.xticks(np.arange(min(raw_train_transaction['addr2']), max(raw_train_transaction['addr2'])+1, 5))

plt.show()
# In order to visualize more deailed insights of add1 variable, we select the top 20 transaction count add1 categories 

top20_addr1 = raw_train_transaction["addr1"].value_counts().head(20).index

top20_addr1 = raw_train_transaction[raw_train_transaction['addr1'].isin(top20_addr1)]
# find out the number of transactions for addr1 categories with the top 20 transaction count 

# find out the fraud transaction percentage of total transactions for addr1 categories with the top 20 transaction count

fig_size=(22,6)

find_percentage(top20_addr1,["addr1"],fig_size,rotation)
# In order to visualize more deailed insights of add1 variable, we selecte the top 20 transaction count add2 categories 

top20_addr2 = raw_train_transaction["addr2"].value_counts().head(20)

top20_addr2 = raw_train_transaction[raw_train_transaction['addr2'].isin(top20_addr2)]
# find out the number of transactions for addr2 categories with the top 20 transaction count 

# find out the fraud transaction percentage of total transactions for addr2 categories with the top 20 transaction count 

find_percentage(top20_addr2,["addr2"],fig_size,rotation)
# select the P_emaildomain categories with the top 20 transaction count

top20_P_email = raw_train_transaction["P_emaildomain"].value_counts().head(20).index

top20_P_email = raw_train_transaction[raw_train_transaction['P_emaildomain'].isin(top20_P_email)]
# find out the number of transactions for P_emaildomain categories with the top 20 transaction count 

# find out the fraud transaction percentage of total transactions for P_emaildomain categories with the top 20 transaction count 

fig_size=(24,6)

find_percentage(top20_P_email,['P_emaildomain'],fig_size,rotation)
# select the R_emaildomain categories with the top 20 transaction count

top20_R_email = raw_train_transaction["R_emaildomain"].value_counts().head(20).index

top20_R_email = raw_train_transaction[raw_train_transaction['R_emaildomain'].isin(top20_R_email)]
# find out the number of transactions for R_emaildomain categories with the top 20 transaction count 

# find out the fraud transaction percentage of total transactions for R_emaildomain categories with the top 20 transaction count 

find_percentage(top20_R_email,['R_emaildomain'],fig_size,rotation)
# create a function to convert TransactionDT variable into a time format

def convert_time (df):

    START_DATE = '2017-01-01'

    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

    df["Date"] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

    df['Weekdays'] = df['Date'].dt.dayofweek

    df['Hours'] = df['Date'].dt.hour

    df['Days'] = df['Date'].dt.day

    df['Month'] = df['Date'].dt.month

    return df
# convert both train and test transaction's TransactionDT variable into a time format

raw_train_transaction = convert_time(raw_train_transaction)

raw_test_transaction =  convert_time(raw_test_transaction)
# plot bar graphs to visualize the number of transaction and fraud transaction percentage for each hour

fig_size=(16,6)

find_percentage(raw_train_transaction,['Hours'],fig_size,rotation)
# plot bar graphs to visualize the number of transaction and fraud transaction percentage for each week day

find_percentage(raw_train_transaction,['Weekdays'],fig_size,rotation)
# plot bar graphs to visualize the number of transaction and fraud transaction percentage for each month

find_percentage(raw_train_transaction,['Month'],fig_size,rotation)
# create a new dataframe to store the number of fraud for each day

fraud_record=raw_train_transaction[raw_train_transaction['isFraud']==1]

daily_fraud=fraud_record.groupby(by=fraud_record['Date'].dt.date).size().reset_index()

daily_fraud.columns=['Date','Number of Fraud']
# create a line graph to visualize the changes in the number of fraud transaction for each day from 2017-01-01 to 2017-07-02

plt.figure(figsize=(16,6))

plt.title("Number of Fraud Transaction per Day")

plt.xlabel("Date")

plt.ylabel("Transaction Count")

plt.plot(daily_fraud['Date'],daily_fraud['Number of Fraud'])

plt.show()
# preview of identify file

raw_train_identity.head()
merged_df = raw_train_transaction.merge(raw_train_identity,how='left', left_index=True, right_index=True)
# select the devices with top 10 transaction count 

top10_Device_Type = merged_df["DeviceType"].value_counts().index

top10_Device_Type  = merged_df[merged_df['DeviceType'].isin(top10_Device_Type)]
# select more detailed devices with top 10 transaction count 

top10_Device_Info = merged_df["DeviceInfo"].value_counts().head(10).index

top10_Device_Info = merged_df[merged_df['DeviceInfo'].isin(top10_Device_Info)]
# plot bar graphs to visualize the number of transaction and fraud transaction percentage for each device  

find_percentage(top10_Device_Type,['DeviceType'],fig_size,rotation)
# plot bar graphs to visualize the number of transaction and fraud transaction percentage for the top 10 devices 

rotation=45

find_percentage(top10_Device_Info,['DeviceInfo'],fig_size,rotation)
# plot bar graphs to visualize other categorical variables in the identity table

rotation=0

id_list= ['id_12','id_15','id_16','id_23','id_27','id_28','id_29','DeviceType']

find_percentage(merged_df,id_list,fig_size,rotation)
# store raw files into a new variable for modification

train_transaction = raw_train_transaction

test_transaction = raw_test_transaction
# log transformation on TransactionAMT variable 

train_transaction['TransactionAmt'] = np.log(raw_train_transaction['TransactionAmt'])

test_transaction['TransactionAmt'] = np.log(raw_test_transaction['TransactionAmt'])
# merge transaction file with identity file 

merged_train_df = train_transaction.merge(raw_train_identity,how='left', left_index=True, right_index=True)

merged_test_df = test_transaction.merge(raw_test_identity,how='left', left_index=True, right_index=True)
# create lists to store categorical variables

v_features = ["V"+str(i) for i in np.arange(1, 340, 1)]

C_cols = ["C"+str(i) for i in np.arange(1, 15, 1)]

card_cols = ["card"+str(i) for i in np.arange(1, 7, 1)]

D_cols = ["D"+str(i) for i in np.arange(1, 16, 1)]

addr_cols = ["addr"+str(i) for i in np.arange(1, 3, 1)]

id_cols = ["id_"+str(i) for i in np.arange(12, 39, 1)]
# create a function to perform pca transformation to reduce the number of variables

def PCA_transform(df, cols,prefix, n_features):

    pca = PCA(n_components = n_features, random_state=101)

    pca_model = pca.fit_transform(df[cols])

    pca_df = pd.DataFrame(pca_model)

    df.drop(cols, axis=1, inplace=True)

    pca_df.rename(columns=lambda x: str(prefix)+str(x), inplace=True)

    df = pd.concat([df, pca_df], axis=1)

    return df
# since pca does not accept NA values, we will fill na with -1 

# before pca transformation the data need to be scaled from 0 to 1 

def fill_na_features (df,features):

    for col in features:

        df[col] = df[col].fillna((df[col].min() - 1))

        df[col] = (minmax_scale(df[col], feature_range=(0,1)))

    return df
# prepare pca transformation

merged_train_df = fill_na_features(merged_train_df,v_features)

merged_test_df = fill_na_features(merged_test_df,v_features)
# perform pca transformation which holds 95% of variance of v_features

merged_train_df = PCA_transform(merged_train_df, v_features, 'PCA_V',20)

merged_test_df = PCA_transform(merged_test_df, v_features, 'PCA_V',20)
cat_cols1 = [card_cols,addr_cols,M_cols,id_cols]

cat_cols2 = ['ProductCD','P_emaildomain','R_emaildomain','DeviceType','DeviceInfo']
# create a function to convert the categorical variable's categories into numbers

def convert_cat_label1(df):

    for i in range(len(cat_cols1)):

        for col in cat_cols1[i]:

            # avoid nan

            if df[col].dtype=='object':

                le = preprocessing.LabelEncoder()

                le.fit(list(df[col].values) + list(df[col].values))

                df[col] = le.transform(list(df[col].values))

    return df
# create a function to convert the categorical variable's categories into numbers

def convert_cat_label2(df):

    for col in cat_cols2:

        if col in df.columns:

            le = preprocessing.LabelEncoder()

            le.fit(list(df[col].values) + list(df[col].values))

            df[col] = le.transform(list(df[col].values))

    return df
# convert categorical variables's categories into numbers 

merged_train_df = convert_cat_label1(merged_train_df)

merged_train_df = convert_cat_label2(merged_train_df)

merged_test_df = convert_cat_label1(merged_test_df)

merged_test_df = convert_cat_label2(merged_test_df)
# collect all the garbabge variables to reduce memory

gc.collect()
# assign indedepnt variables to X, and depdent variable isFraud to y

X= merged_train_df.drop(['TransactionID_x','TransactionDT','isFraud','Date'],axis=1)

y=merged_train_df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_submission = merged_test_df.drop(['TransactionID_x','TransactionDT','Date'],axis=1)
# use hypteropt to optimize the parameters of xgb classifier 

'''

def objective(params):

    params = {

        'max_depth': int(params['max_depth']),

        'subsample':'{:.3f}'.format(params['subsample']),

        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),

    }

    

    clf = xgb.XGBClassifier(

        n_estimators=500,

        learning_rate=0.05,

        n_jobs=4,

        random_state=101

        **params

    )



    #score = cross_val_score(clf, X_test, y_test, scoring='roc_auc', cv=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf.fit(X_train, y_train)

    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])



    print("auc {:.3f} params {}".format(auc, params))

    return auc



space = {

    'max_depth': hp.quniform('max_depth', 3, 8, 1),

    'subsample': hp.uniform('subsample', 0.6, 0.9),

    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 0.9),

}

'''
'''

%%time

best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=5)

print("Hyperopt estimated optimum {}".format(best))

'''
# use xgboost to classify whether each transaction is fraud or not

import xgboost as xgb

clf = xgb.XGBClassifier( n_estimators=500,

    max_depth=7,

    learning_rate=0.05,

    subsample=0.9,

    colsample_bytree=0.9,

    random_state=101)
# fit the model with x label and y label and predict X_test

clf.fit(X_train,y_train)

y_preds = clf.predict_proba(X_test)
# accuracy on y_test 

auc = roc_auc_score(y_test, y_preds[:,1])

print('AUC: %.3f' % auc)
# predict the probability of each transaction is fraud or not

y_preds = clf.predict_proba(X_submission)
# merge prediction results with test transactions

sample_submission = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")

sample_submission['isFraud']=y_preds[:,1]

sample_submission.head(10)

sample_submission.to_csv('final_result.csv', index=False)
# display the first 10 rows

sample_submission.head(10)