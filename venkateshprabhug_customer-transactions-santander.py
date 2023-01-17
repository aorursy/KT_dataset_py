import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
import gc

import os

import warnings

warnings.filterwarnings('ignore')
pd.options.display.precision = 2

pd.options.display.max_columns = 250

pd.set_option('float_format', '{:2f}'.format)

seed = 10
print(os.listdir('../input'))
path = '../input/santander-customer-transaction-prediction'
tr = pd.read_csv(f'{path}/train.csv')

ts = pd.read_csv(f'{path}/test.csv')
tr.head(3)
ts.head(3)
print(f'Train shape: {tr.shape}')

print(f'Test shape: {ts.shape}')
print(f'Are there any missing values in train? {tr.isnull().sum().any()}')

print(f'Are there any missing values in test? {ts.isnull().sum().any()}')
sns.countplot(tr['target'])

plt.title(f'Positive class: {round(tr["target"].value_counts()[1]/len(tr) * 100, 2)}%')

plt.show()
# Create correlation matrix

corr = tr.corr()['target'][1:].abs()

correlations = pd.DataFrame({'column': corr.index, 'correlation': corr}).sort_values('correlation', ascending=False).reset_index(drop=True)

correlations.head()
plt.figure(figsize=(15, 5))

plt.plot(correlations['column'][:20], correlations['correlation'][:20])

plt.xticks(correlations['column'][:20], correlations['column'][:20], rotation='45')

plt.title('Feature Correlations')

plt.show()
fig = plt.figure(figsize = (10,10))

ax = fig.gca()

cols = correlations['column'][:10].values

tr[cols].hist(ax = ax)

plt.show()
from sklearn.ensemble import RandomForestClassifier
base_model = RandomForestClassifier(random_state=seed, class_weight={0:1, 1:9}, n_estimators=20, verbose=0)

%time base_model.fit(tr.drop(['ID_code', 'target'], 1), tr['target'])
importances = pd.DataFrame({'feature': tr.drop(['ID_code', 'target'], 1).columns, 'importance': base_model.feature_importances_}).sort_values('importance', ascending=False).reset_index(drop=True)

importances[:10]
top = 100

selected_features = importances['feature'][:top].values

print(selected_features)
tr = tr.sample(random_state=seed, frac=1)
features = tr[selected_features]

target = tr['target']



# Payload here represents the actual test data for which we are trying to predict in this challenge

payload = ts[selected_features]
features.head()
features.shape, payload.shape, target.shape
# from sklearn.decomposition import PCA

# # from MulticoreTSNE import MulticoreTSNE as TSNE
# decomposed = PCA(n_components=100).fit_transform(features)
from sklearn.model_selection import train_test_split



x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=seed, stratify=target)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier
model = CatBoostClassifier(random_state=seed, 

                           scale_pos_weight=10, 

                           silent=True, 

                           max_depth=None, 

                           learning_rate=0.2, 

                           loss_function='Logloss', 

                           n_estimators=2000)

%time model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=50, plot=True)
y_train_pred = model.predict(x_train)

y_val_pred = model.predict(x_val)
from sklearn.metrics import confusion_matrix, roc_auc_score



pd.DataFrame({'Train Set': roc_auc_score(y_train, y_train_pred)*100, 'Validation Set': roc_auc_score(y_val, y_val_pred)*100}, index=['ROC'])
plt.figure(figsize=(5, 3))

a, b = np.bincount(y_val)

# sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='g')

# plt.show()

sns.heatmap(np.stack([(confusion_matrix(y_val, y_val_pred)[0]/a)*100, (confusion_matrix(y_val, y_val_pred)[1]/b)*100], 0), annot=True, fmt='g')

plt.show()
predictions = model.predict(payload).flatten()



submission = pd.DataFrame({'ID_code': ts['ID_code'], 'target': predictions})



submission.to_csv('submission.csv', index=False)