from datetime import timedelta
import pandas as pd
marketing_data_df = pd.read_csv('/kaggle/input/sampledataset/marketing.csv')
transaction_data_df = pd.read_csv("/kaggle/input/sampledataset/transaction.csv")
pricing_data_df = pd.read_csv("/kaggle/input/sampledataset/pricing.csv")
activity_data_df = pd.read_csv("/kaggle/input/sampledataset/activity.csv")
activity_data_df.ACTIVITY.value_counts()
new_users_data = pd.read_csv("/kaggle/input/sampledataset2/problem-two-new-users.csv",header=None)[0].tolist()
len(new_users_data)
activity_data_df.DATE = pd.to_datetime(activity_data_df.DATE, format='%Y%m%d')
register_data_df = activity_data_df[['DATE',"USERID"]][activity_data_df.ACTIVITY == 'created account']
register_data_df['A_MONTH_SINCE_REGISTER'] = register_data_df.DATE + timedelta(days=28)
register_data_df.rename(columns={"DATE":"REGISTER_DATE"},inplace=True)
register_data_df.head()
register_data_df.USERID.isin(new_users_data).sum()
activity_data_df = pd.merge(activity_data_df, register_data_df, on='USERID')
new_activity_data_df = activity_data_df[(activity_data_df.DATE >= activity_data_df.REGISTER_DATE) & (activity_data_df.DATE <= activity_data_df.A_MONTH_SINCE_REGISTER)]
logged_10times_since = new_activity_data_df.sort_values(by='DATE').groupby("USERID").head(11)
len(set.intersection(set(new_activity_data_df.USERID.unique().tolist()),set(new_users_data)))
new_activity_data_df.head()
logged_10times_since = logged_10times_since.groupby("USERID").DATE.max().reset_index()
logged_10times_since.rename(columns = {"DATE":"LOGGED_10TIMES_SINCE"},inplace=True)
num_logins = activity_data_df[activity_data_df.ACTIVITY=='logged in'].groupby("USERID").DATE.count().reset_index()
num_logins.rename(columns = {"DATE":"NUM_LOGINS"},inplace=True)
logged_10times_since = pd.merge(logged_10times_since, num_logins, on='USERID')
logged_10times_since = logged_10times_since[logged_10times_since.NUM_LOGINS >= 10]
new_activity_data_df = pd.merge(new_activity_data_df, logged_10times_since,on='USERID',how='left')
new_activity_data_df = new_activity_data_df[new_activity_data_df.LOGGED_10TIMES_SINCE <= new_activity_data_df.A_MONTH_SINCE_REGISTER]
df= new_activity_data_df[['USERID',"LOGGED_10TIMES_SINCE",'A_MONTH_SINCE_REGISTER','REGISTER_DATE','NUM_LOGINS']].drop_duplicates()
df['IS_TEST'] = df.USERID.isin(new_users_data)
df = df[df.NUM_LOGINS>=10]
import numpy as np
df['INTERVAL'] = (df.A_MONTH_SINCE_REGISTER - df.LOGGED_10TIMES_SINCE).dt.days
df['RANDOM_INTERVAL'] = df.INTERVAL.map(lambda x: np.random.randint(0,x+1))
df['EVALUATE_FROM'] = df.LOGGED_10TIMES_SINCE + df.RANDOM_INTERVAL * timedelta(days=1)
df.loc[df.IS_TEST,'EVALUATE_FROM'] = pd.to_datetime('20160703',format='%Y%m%d')
df.drop(['INTERVAL','RANDOM_INTERVAL','NUM_LOGINS','A_MONTH_SINCE_REGISTER'],axis=1,inplace=True)
transaction_data_df = pd.merge(transaction_data_df, df, on='USERID',how='inner')
transaction_data_df.DATE = pd.to_datetime(transaction_data_df.DATE, format='%Y%m%d')
label_data_df = transaction_data_df[(transaction_data_df.DATE >= transaction_data_df.EVALUATE_FROM) & (transaction_data_df.DATE <=transaction_data_df.EVALUATE_FROM + timedelta(days=91))]
label_data_df = label_data_df.groupby("USERID").TOTAL.sum()
label_data_df=label_data_df.reset_index().rename(columns = {"TOTAL":"FUTURE_AMOUNT"})
df = pd.merge(df,label_data_df, on='USERID',how='left').fillna(0)
df['DAYS_SINCE_REGISTER'] = (df.EVALUATE_FROM - df.REGISTER_DATE).dt.days
historical_amount = transaction_df[transaction_data_df.DATE < transaction_data_df.EVALUATE_FROM]
historical_amount = historical_amount.groupby("USERID").agg({"TOTAL":['sum','mean','median','max','min','count']}).reset_index()
historical_amount.columns = ["USERID",*['historical_amount_' + i for i in ['sum','mean','median','max','min','count']]]
df = pd.merge(df, historical_amount, on='USERID',how='left')
activity_data_df.DATE = pd.to_datetime(activity_data_df.DATE, format='%Y%m%d')
login_activity = activity_data_df[activity_data_df.ACTIVITY=='logged in']
login_activity = pd.merge(login_activity, df[['USERID','REGISTER_DATE','EVALUATE_FROM']],on='USERID')
login_activity = login_activity[login_activity.DATE < login_activity.EVALUATE_FROM]
login_activity['LAST_LOGGED_DATE'] =login_activity.groupby("USERID").DATE.shift()
login_activity['LOG_INTERVAL'] = (login_activity.DATE - login_activity.LAST_LOGGED_DATE).dt.days
historical_login_activity = login_activity.groupby("USERID").agg({"LOG_INTERVAL":['sum','count','mean','min','max','median']}).reset_index()
historical_login_activity.columns =  ["USERID",*['log_interval_' + i for i in ['sum','count','mean','min','max','median']]]
df = pd.merge(df, historical_login_activity, on='USERID',how='left').fillna(0)
df.set_index("USERID",inplace=True)
df.plot(x='REGISTER_DATE',y='FUTURE_AMOUNT')
transaction_df.DATE.max()
train_df = df[~df.IS_TEST]
train_df[train_df.REGISTER_DATE <= '2016-03-02']
test_df = df[df.IS_TEST]

train_df.drop(['LOGGED_10TIMES_SINCE','REGISTER_DATE','EVALUATE_FROM',"IS_TEST"],axis=1,inplace=True)
test_df.drop(['LOGGED_10TIMES_SINCE','REGISTER_DATE','EVALUATE_FROM','IS_TEST'],axis=1,inplace=True)
train_df['label'] = train_df.FUTURE_AMOUNT >= 100
from sklearn.model_selection import train_test_split
train, valid = train_test_split(train_df,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
train.label.value_counts()
clf = RandomForestClassifier(n_estimators=500,n_jobs=4,class_weight=None)
X_train = train.drop(['label','FUTURE_AMOUNT'],axis=1)
y_train = train['label']
X_valid = valid.drop(['label','FUTURE_AMOUNT'],axis=1)
y_valid = valid['label']
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
clf.fit(X_train, y_train)
y_valid_pred = clf.predict(X_valid)
from sklearn.metrics import confusion_matrix, precision_score, recall_score,accuracy_score
confusion_matrix(y_valid, y_valid_pred)
precision_score(y_valid, y_valid_pred)
recall_score(y_valid, y_valid_pred)
clf = RandomForestClassifier(n_estimators=500,n_jobs=4,class_weight=None)
clf.fit(pd.concat([X_train,X_valid]), pd.concat([y_train,y_valid]))
pred = clf.predict(test_df.drop(['FUTURE_AMOUNT'],axis=1))