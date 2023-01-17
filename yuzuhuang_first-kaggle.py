# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# load data
df_train = pd.read_csv('../input/application_train.csv')
print('Training data shape:{}'.format(df_train.shape))
df_train.head(10)
df_test = pd.read_csv('../input/application_test.csv')
print('Testing data shape:{}'.format(df_test.shape))
df_test.head(10)
sns.distplot(df_train['TARGET'], kde = False)
#Missing value
missing_value = df_train.isnull().sum().sort_values(ascending=False)
missing_value.head(20)
#label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df_train:
    if df_train[col].dtype == 'object':
        if len(list(df_train[col].unique()))<=2:
            le.fit(df_train[col])
            df_train[col] = le.transform(df_train[col])
            df_test[col] = le.transform(df_test[col])
#one-hot encoding
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)
print('Training data shape:{}'.format(df_train.shape))
print('Testing data shape:{}'.format(df_test.shape))
x = df_train['TARGET']
df_train, df_test = df_train.align(df_test, join='inner', axis=1)
df_train['TARGET'] = x
print('Training data shape:{}'.format(df_train.shape))
print('Testing data shape:{}'.format(df_test.shape))
correlations = df_train.corr()['TARGET'].sort_values()
correlations.head(10)
correlations.tail(10)
ext_train = df_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
ext_train.correlations = ext_train.corr()
ext_train.correlations
plt.figure(figsize = (8, 6))
sns.heatmap(ext_train.correlations, vmin = -0.5, annot = True, vmax = 0.5)
plt.figure(figsize = (8,6))
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    plt.subplot(3, 1, i + 1)
    sns.kdeplot(ext_train.loc[ext_train['TARGET'] == 0, source], label = 'TARGET == 0')
    sns.kdeplot(ext_train.loc[ext_train['TARGET'] == 1, source], label = 'TARGET == 1')
    plt.title('Distribution of {} by TARGET value'.format(source))
plt.tight_layout(h_pad = 2.5)
for col in ext_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]:
    col_median_1 = ext_train[col].median()
    ext_train[col] = ext_train[col].fillna(col_median_1)
ext_train.isnull().sum()
ext_test = df_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
for col in ext_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]:
    col_median_2 = ext_test[col].median()
    ext_test[col] = ext_test[col].fillna(col_median_2)
ext_test.isnull().sum()
X = ext_train.drop(['TARGET'], axis = 1)
y = ext_train['TARGET']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
RF = RandomForestClassifier(random_state = 1, n_estimators = 100, min_samples_split = 4, min_samples_leaf = 2)
RF.fit(X_train, y_train)
y_pred_1 = RF.predict(X_test)
predictions_1 = [value for value in y_pred_1]
accuracy = accuracy_score(y_test, predictions_1)
print('The accuracy of RF model is {}'.format(accuracy))
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_2 = xgb.predict(X_test)
predictions_2 = [value for value in y_pred_2]
accuracy = accuracy_score(y_test, predictions_2)
print('The accuracy of xgboost model is {}'.format(accuracy))
xgb_pred = xgb.predict_proba(ext_test)[:,1]
xgb_pred
final = df_test[['SK_ID_CURR']]
final['TARGET'] = xgb_pred
final.head()
final.to_csv('/Users/huangyuzu/Desktop/python/xgboost_result.csv', index = False)