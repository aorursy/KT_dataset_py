#import libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import pandas_profiling

import seaborn as sns

import gc

from scipy import stats
#importing IEEE dataset

IEEE_data = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

IEEE_Identity_data = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

IEEE_test_data = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

IEEE_Identity_test_data = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
pd.set_option('display.max_columns', None)

IEEE_data.head()


IEEE_Identity_data.head()
IEEE_test_data.head()
IEEE_Identity_test_data.head()
IEEE_data.shape
IEEE_Identity_data.shape
IEEE_test_data.shape
IEEE_Identity_test_data.shape
# Merging datasets based on TransactionID

IEEE_train = pd.merge(IEEE_data, IEEE_Identity_data, on = 'TransactionID', how = 'left')

IEEE_test = pd.merge(IEEE_test_data, IEEE_Identity_test_data, on = 'TransactionID', how = 'left')
del IEEE_data, IEEE_Identity_data, IEEE_test_data, IEEE_Identity_test_data

gc.collect()
IEEE_train.shape
IEEE_test.shape


pd.set_option('display.max_rows', None)

IEEE_train.head(100)
IEEE_test.head()
IEEE_train.describe()
def resumetable(df):

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary



## Function to reduce the DF size

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



def CalcOutliers(df_num): 



    # calculating mean and std of the array

    data_mean, data_std = np.mean(df_num), np.std(df_num)



    # seting the cut line to both higher and lower values

    # You can change this value

    cut = data_std * 3



    #Calculating the higher and lower cut values

    lower, upper = data_mean - cut, data_mean + cut



    # creating an array of lower, higher and total outlier values 

    outliers_lower = [x for x in df_num if x < lower]

    outliers_higher = [x for x in df_num if x > upper]

    outliers_total = [x for x in df_num if x < lower or x > upper]



    # array without outlier values

    outliers_removed = [x for x in df_num if x > lower and x < upper]

    

    print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers

    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers

    print('Total outlier observations: %d' % len(outliers_total)) # printing total number of values outliers of both sides

    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values

    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points

    

    return
resumetable(IEEE_train)
#Outliers in Training Data

CalcOutliers(IEEE_train['TransactionAmt'])
# Number of Columns having Null values in train data

print(IEEE_train.isnull().any().sum())
# Number of Columns having Null in Test Data

print(IEEE_test.isnull().any().sum())
#Taking Sample Data to Perform data visualization

Sample_train = IEEE_train.head(1000)
# Transaction amount for different service providers

sns.catplot(x="card4", y="TransactionAmt", data=Sample_train, height = 8, aspect = 1.5)

plt.show()
# Credit and Debit Transaction amount

sns.catplot(x="card6", y="TransactionAmt", data=Sample_train, height = 8, aspect = 1.5)

plt.show()
sns.relplot(x="TransactionAmt", y="card2", hue="card4", style = "card6", data=Sample_train, height = 8, aspect = 1.5);
#Fraud data through diffrent cards

sns.catplot(x="isFraud", y="TransactionAmt", hue="card4", kind="swarm", data=Sample_train, height = 8, aspect = 1.5)

plt.show()
sns.catplot(x="isFraud", y="TransactionAmt", hue="card6", kind="swarm", data=Sample_train, height = 8, aspect = 1.5)

plt.show()
# Credit and Debit amount through diffrent service providers

sns.catplot(x="card4", y="TransactionAmt", hue="card6", kind="swarm", data=Sample_train, height = 8, aspect = 1.5)

plt.show()
sns.catplot(x="DeviceType", y="TransactionAmt", kind="boxen",

            data=Sample_train.sort_values("isFraud"), height = 8, aspect = 1.5)

plt.show()
# Fraud transactions using Different devices

sns.catplot(x="isFraud", y="TransactionAmt", hue="DeviceType",

            kind="violin", inner="stick", split=True,

            palette="pastel", data=Sample_train, height = 8, aspect = 1.5)

plt.show()
#Relationship between card1 and Transaction amount

sns.lmplot(x="card1", y="TransactionAmt", col="card4", data=Sample_train,

           aspect=1.0);
# Use of Mobile and Desktop devices for service providers

sns.catplot(x="DeviceType", y="TransactionAmt", hue="card4", kind="bar", data=Sample_train, height = 8, aspect = 1.5)

plt.show()
# Fraud by Mobile and Desktop trasactions 

sns.catplot(x="isFraud", y="TransactionAmt", hue="card4",

            col="DeviceType",height = 8, aspect=1.0,

            kind="bar", data=Sample_train)

plt.show()
# Found and Not Found Transaction

sns.catplot(x="isFraud", y="TransactionAmt", hue="id_12",

            col="DeviceType",height = 8, aspect=1.0,

            kind="bar", data=Sample_train)

plt.show()
# Credit and Debit transactions where fraud has found or not

sns.relplot(x="isFraud", y="TransactionAmt", hue="card6",

            col="id_12", row="DeviceType",

            kind="line", estimator=None, data=Sample_train, height = 6, aspect=1.0);
#Reducing Memory uses

IEEE_train = reduce_mem_usage(IEEE_train)

IEEE_test = reduce_mem_usage(IEEE_test)
ID = IEEE_test.TransactionID
#column wise Number of Null values in training Data

IEEE_train.isna().sum()
#Showing Categories in Categorical train Data

print (IEEE_train['M6'].unique())

print (IEEE_train['card4'].unique())

print (IEEE_train['ProductCD'].unique())

print (IEEE_train['id_12'].unique())

print (IEEE_train['DeviceType'].unique())
#Showing Categories in Categorical test Data

print (IEEE_test['M6'].unique())

print (IEEE_test['card4'].unique())

print (IEEE_test['ProductCD'].unique())

print (IEEE_test['id-12'].unique())

print (IEEE_test['DeviceType'].unique())
# Handling Null value in Categorical Field

def impute_nan_create_category(DataFrame,ColName):

    DataFrame.loc[:, ColName]=np.where(DataFrame.loc[:, ColName].isnull(),"Unknown",DataFrame.loc[:, ColName])
for Columns in ['card4','M6']:

    impute_nan_create_category(IEEE_train,('card4', 'M6'))
for Columns in ['id_12', 'DeviceType']:

    impute_nan_create_category(IEEE_train,('id_12','DeviceType'))
X_train = IEEE_train.iloc[:, np.r_[3:10,11:13,17:32, 40:41, 45:46,51:52,66:89, 333:376,405:406, 432:433]].values
Y_train = IEEE_train.iloc[:, 1].values
del IEEE_train

gc.collect()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X_train[:, np.r_[3:5,6:9,23:26,27:93]])

X_train[:, np.r_[3:5,6:9,23:26,27:93]] = imputer.transform(X_train[:, np.r_[3:5,6:9,23:26,27:93]])
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,5,26,93,94])], remainder='passthrough')

X_train = np.array(ct.fit_transform(X_train))
#column wise Number of Null values in test Data

IEEE_test.isna().sum()
for Columns in ['card4','M6']:

    impute_nan_create_category(IEEE_test,('card4', 'M6'))
for Columns in ['id_12', 'DeviceType']:

    impute_nan_create_category(IEEE_test,('id-12', 'DeviceType'))
X_test = IEEE_test.iloc[:, np.r_[2:9,10:12,16:31, 39:40, 44:45,50:51,65:88, 332:375,404:405, 431:432]].values
del IEEE_test

gc.collect()
imputer.fit(X_train[:, np.r_[3:5,6:26,27:93]])

X_test[:, np.r_[3:5,6:26,27:93]] = imputer.transform(X_test[:, np.r_[3:5,6:26,27:93]])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,5,26,93,94])], remainder='passthrough')

X_test = np.array(ct.fit_transform(X_test))
X_train.shape
X_test.shape
# Applying Feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#Applying PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = 3)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
from xgboost import XGBClassifier

classifier = XGBClassifier()

Xg_model = classifier.fit(X_train, Y_train)
Fraud = Xg_model.predict(X_test)
#Submitting the Output

Fraud_prediction = pd.DataFrame({'TransactionID': ID, 'isFraud': Fraud})
Fraud_prediction.to_csv('kapilv_IEEE_submission1.csv', index=False)