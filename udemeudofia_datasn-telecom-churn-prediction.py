import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font",size=14)
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
raw_data = pd.read_csv('../input/TRAIN.csv').dropna(axis=0, how='all')
print(raw_data.shape)
print(list(raw_data.columns))
raw_data.head(20)
raw_data['Churn Status'].dtype
raw_data['Most Loved Competitor network in in Month 1'].unique()
raw_data['Most Loved Competitor network in in Month 2'].unique()
raw_data['Network type subscription in Month 1'].unique()
raw_data['Network type subscription in Month 2'].unique()
raw_data[raw_data['Most Loved Competitor network in in Month 1']=='0']
raw_data['Churn Status'].value_counts()
sns.countplot(x=raw_data['Churn Status'], data=raw_data, palette='hls')
plt.show()
raw_data.groupby('Churn Status').mean()
raw_data.groupby('Most Loved Competitor network in in Month 1').mean()
raw_data.groupby('Most Loved Competitor network in in Month 2').mean()
raw_data.groupby('Network type subscription in Month 1').mean()
raw_data.groupby('Network type subscription in Month 2').mean()
pd.crosstab(raw_data['Most Loved Competitor network in in Month 1'], raw_data['Churn Status']).plot(kind='bar')
plt.title('Churn Status for Most loved in Month 1')
plt.xlabel('Most loved in Month 1')
plt.ylabel('Frequency of Churn')
plt.savefig('churn_fre_mostlovedmonth1')
pd.crosstab(raw_data['Most Loved Competitor network in in Month 2'], raw_data['Churn Status']).plot(kind='bar')
plt.title('Churn Status for Most loved in Month 2')
plt.xlabel('Most loved in Month 2')
plt.ylabel('Frequency of Churn')
plt.savefig('churn_fre_mostlovedmonth2')
pd.crosstab(raw_data['Network type subscription in Month 1'], raw_data['Churn Status']).plot(kind='bar')
plt.title('Churn Status for Network type in Month 1')
plt.xlabel('Network type in Month 1')
plt.ylabel('Frequency of Churn')
plt.savefig('churn_fre_networktypemonth1')
pd.crosstab(raw_data['Network type subscription in Month 2'], raw_data['Churn Status']).plot(kind='bar')
plt.title('Churn Status for Network type in Month 2')
plt.xlabel('Network type in Month 2')
plt.ylabel('Frequency of Churn')
plt.savefig('churn_fre_networktypemonth2')
raw_data = raw_data.drop(['Network type subscription in Month 2'], axis=1)
raw_data[raw_data['Most Loved Competitor network in in Month 1']=='0']
raw_data = pd.get_dummies(raw_data,columns=['Network type subscription in Month 1', 'Most Loved Competitor network in in Month 1','Most Loved Competitor network in in Month 2'])
raw_data.columns.values
y = raw_data['Churn Status']
X = raw_data.drop(['Customer ID','Churn Status','Most Loved Competitor network in in Month 1_0'],axis=1)
X.columns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 18)
rfe = rfe.fit(X, y)
print(rfe.support_)
print(rfe.ranking_)
X_data = X.drop(['network_age','Customer tenure in month','Total Spend in Months 1 and 2 of 2017','Total Data Spend','Total Data Consumption', 'Total Onnet spend ','Total Offnet spend'],axis=1)
import statsmodels.api as sm
logit_model=sm.Logit(y,X_data)
result=logit_model.fit()
print(result.summary())
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.3, random_state=0)
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
xg_cl = xgb.XGBClassifier(objective='binary:logistic', max_depth=5, learning_rate=1.91, silent=1.0, n_estimators=5)
xg_cl.fit(X_train,y_train)
y_pred = xg_cl.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(xg_cl.score(X_test, y_test)))
test_data = pd.read_csv('../input/TEST.csv').dropna(axis=0, how='all')
print(test_data.shape)
print(list(test_data.columns))
xgbm = xgb.XGBClassifier(
   learning_rate=0.02,
   n_estimators=1500,
   max_depth=6,
   min_child_weight=1,
   gamma=0,
   subsample=0.9,
   colsample_bytree=0.85,
   objective= 'binary:logistic',
   nthread=4,
   scale_pos_weight=1,
   seed=27)

alg = modelfit(xgbm, X_train, y_train)
dtrain_predprob = alg.predict_proba(X_test,y_test)
test_data.info()
test_data.head(20)
test_data = test_data.drop(['Network type subscription in Month 2'], axis=1)
test_data = pd.get_dummies(test_data,columns=['Network type subscription in Month 1', 'Most Loved Competitor network in in Month 1','Most Loved Competitor network in in Month 2'])
test_data.columns
X_result = test_data.drop(['Customer ID', 'network_age','Customer tenure in month','Total Spend in Months 1 and 2 of 2017','Total Data Spend','Total Data Consumption', 'Most Loved Competitor network in in Month 1_0', 'Total Onnet spend ','Total Offnet spend'],axis=1)
predictions = xg_cl.predict(X_result)
cust_id = test_data['Customer ID']
df = pd.DataFrame(cust_id, columns=['Customer ID'])
df['Churn Status'] = predictions.astype(np.int64)
df['Churn Status'].value_counts()
sns.countplot(x=df['Churn Status'], data=df, palette='hls')
plt.show()
df['Churn Status'].dtype
df['Churn Status'].dtype
df.info()
df.to_csv('new_predictions1.csv', index=False)
