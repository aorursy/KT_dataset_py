import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



from scipy.stats import boxcox

from sklearn.preprocessing import RobustScaler, power_transform

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 100)

warnings.filterwarnings('ignore')
# Loading dataset into memory

df = pd.read_csv('../input/creditcardfraud/creditcard.csv')



# Shape of dataset

print(df.shape)



# First 5 rows of dataset

df.head()
# Basic statistical parameters of each column in dataset

df.describe()
# Select all column names except "Class"

cols = df.drop('Class', axis = 1).columns



# Histogram plots of selected columns

df[cols].hist(figsize = (20, 10), bins = 30)

plt.tight_layout()

plt.show()
# Scaling Amount and Time columns

scaler = RobustScaler()

df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))



df.drop(['Amount', 'Time'], axis = 1, inplace = True)
# Plot scaled Amount and Time features

df[['Amount_scaled', 'Time_scaled']].hist(figsize = (20, 5), bins = 50)

plt.show()
def skew_kurt(dataset):

    '''

    This function returns dataset, which contains skew and kurtosis for each of dataset columns except "Class"

    '''

    cols = dataset.drop('Class', axis = 1).columns



    skew = dataset[cols].skew()

    kurtosis = dataset[cols].kurtosis()



    return pd.DataFrame({'Skew': skew.values, 'Kurtosis': kurtosis.values}, index = skew.index).T
# Skew and kurtosis of original dataset

orig_skew = skew_kurt(df)

orig_skew.head()
# Applying 'yeo-johnson' transformation to transformed dataset

df_transformed = df.copy()

cols = df_transformed.drop('Class', axis = 1).columns



for col in cols:

    if orig_skew.loc['Skew', col] < -1 or orig_skew.loc['Skew', col] > 1:        

        df_transformed[col] = power_transform(df[col].values.reshape(-1, 1), method = 'yeo-johnson')
# Histograms of transformed features

df_transformed[cols].hist(figsize = (15, 8), bins = 30)

plt.tight_layout()

plt.show()
# Comparison of skew and kurtosis of original and transformed dataset

transformed_skew = skew_kurt(df_transformed)



plt.figure(figsize = (20, 6))



plt.subplot(121)

plt.plot(orig_skew.T[['Skew']], label = 'orig_skew')

plt.plot(transformed_skew.T[['Skew']], label = 'trans_skew')

plt.legend(); plt.grid(); plt.xticks(rotation = 90)

plt.title('Skewness before and after transformation'); plt.xlabel('Column'); plt.ylabel('Skew')



plt.subplot(122)

plt.plot(orig_skew.T[['Kurtosis']], label = 'orig_kurt')

plt.plot(transformed_skew.T[['Kurtosis']], label = 'trans_kurt')

plt.legend(); plt.grid(); plt.xticks(rotation = 90)

plt.title('Kurtosis before and after transformation'); plt.xlabel('Column'); plt.ylabel('Kurtosis')

plt.show()
# Defining training data and labels

cols = df.drop('Class', axis = 1).columns

X_orig = df[cols]

X_trans = df_transformed[cols]

Y = df['Class']



# Split training data using 33% of data as test dataset

train_x_orig, test_x_orig, train_y_orig, test_y_orig = train_test_split(X_orig, Y, test_size = 0.33, stratify = Y, random_state = 1)

train_x_trans, test_x_trans, train_y_trans, test_y_trans = train_test_split(X_trans, Y, test_size = 0.33, stratify = Y, random_state = 1)
# Training IsolationForest algorithm for both datasets

clf_orig = IsolationForest(n_estimators = 200, max_samples = 1.0, n_jobs = -1, verbose = 1, random_state = 1)

clf_orig.fit(train_x_orig)

preds_orig = clf_orig.predict(test_x_orig)



clf_trans = IsolationForest(n_estimators = 200, max_samples = 1.0, n_jobs = -1, verbose = 1, random_state = 1)

clf_trans.fit(train_x_trans)

preds_trans = clf_trans.predict(test_x_trans)
# Plot confusion matrix

fig = plt.figure(figsize = (10, 10))



preds_orig = np.where(preds_orig == -1, 1, 0)

preds_trans = np.where(preds_trans == -1, 1, 0)



plt.subplot(121)

confusion_orig = confusion_matrix(test_y_orig, preds_orig)

sns.heatmap(confusion_orig, annot = True, fmt = 'd', square = True, xticklabels =  ['P_Non_fraud', 'P_Fraud'], 

            yticklabels = ['Non_fraud', 'Fraud'], cbar = False, cmap = 'Blues').set_title('Original dataset')



plt.subplot(122)

confusion_trans = confusion_matrix(test_y_trans, preds_trans)

sns.heatmap(confusion_trans, annot = True, fmt = 'd', square = True, xticklabels =  ['P_Non_fraud', 'P_Fraud'], 

            yticklabels = ['Non_fraud', 'Fraud'], cbar = False, cmap = 'Blues').set_title('Transformed dataset')

plt.show()
# To test LOF I'll use only 10% of the data, because neighbouring classifiers can be eally slow and memory consuming with big amount of samples 

train_x_orig, test_x_orig, train_y_orig, test_y_orig = train_test_split(X_orig, Y, test_size = 0.1, stratify = Y, random_state = 1)

train_x_trans, test_x_trans, train_y_trans, test_y_trans = train_test_split(X_trans, Y, test_size = 0.1, stratify = Y, random_state = 1)



lof_orig = LocalOutlierFactor(n_jobs = -1)

lof_preds_orig = lof_orig.fit_predict(test_x_orig)



lof_trans = LocalOutlierFactor(n_jobs = -1)

lof_preds_trans = lof_trans.fit_predict(test_x_trans)
fig = plt.figure(figsize = (10, 10))



lof_preds_orig = np.where(lof_preds_orig == -1, 1, 0)

lof_preds_trans = np.where(lof_preds_trans == -1, 1, 0)



plt.subplot(121)

confusion_orig = confusion_matrix(test_y_orig, lof_preds_orig)

sns.heatmap(confusion_orig, annot = True, fmt = 'd', square = True, xticklabels =  ['P_Non_fraud', 'P_Fraud'], 

            yticklabels = ['Non_fraud', 'Fraud'], cbar = False, cmap = 'Blues').set_title('Original dataset')



plt.subplot(122)

confusion_trans = confusion_matrix(test_y_trans, lof_preds_trans)

sns.heatmap(confusion_trans, annot = True, fmt = 'd', square = True, xticklabels =  ['P_Non_fraud', 'P_Fraud'], 

            yticklabels = ['Non_fraud', 'Fraud'], cbar = False, cmap = 'Blues').set_title('Transformed dataset')

plt.show()