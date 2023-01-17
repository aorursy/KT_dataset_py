# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
df = pd.read_csv("../input/random-forest-hotel-booking/hotel_bookings.csv")

df.head()
df.shape
df.dtypes
df.describe()
df["is_canceled"].value_counts()
df["hotel"].value_counts()
df["target"] = np.where(df["is_canceled"].isin(["1"]), 1, 0)

df["target"].value_counts()
df["target"].mean()
sns.countplot(x = "target", data = df)

plt.show
## checking missing values

df.isnull().sum()
# Replace missing values:

# agent: If no agency is given, booking was most likely made without one.

# company: If none given, it was most likely private.

# rest schould be self-explanatory.

nan_replacements = {"children": 0,"country": "Unknown", "agent": 0, "company": 0}

df = df.fillna(nan_replacements)



# "meal" contains values "Undefined", which is equal to SC.

df["meal"].replace("Undefined", "SC", inplace=True)
## Checking null values

df.isnull().sum()
ax = sns.barplot(x = "meal", y = "target", data = df, estimator = np.mean)
ax = sns.barplot(x = "arrival_date_month", y = "target", data = df, estimator = np.mean)
df.groupby("arrival_date_month")["target"].mean()
df.groupby("country")["target"].mean()
ax = sns.barplot(x = "market_segment", y = "target", data = df, estimator = np.mean)
df.groupby("market_segment")["target"].count()
ax = sns.barplot(x = "distribution_channel", y = "target", data = df, estimator = np.mean)
df.groupby("distribution_channel")["target"].count()
ax = sns.barplot(x = "reserved_room_type", y = "target", data = df, estimator = np.mean)
df.groupby("reserved_room_type")["target"].count()
ax = sns.barplot(x = "assigned_room_type", y = "target", data = df, estimator = np.mean)
df.groupby("assigned_room_type")["target"].count()
ax = sns.barplot(x = "deposit_type", y = "target", data = df, estimator = np.mean)
df.groupby("deposit_type")["target"].count()
ax = sns.barplot(x = "customer_type", y = "target", data = df, estimator = np.mean)
df.groupby("customer_type")["target"].count()
ax = sns.barplot(x = "reservation_status", y = "target", data = df, estimator = np.mean)

df.groupby("reservation_status")["target"].count()
ax = sns.barplot(x = "lead_time", y = "target", data = df, estimator = np.mean)
df.groupby("lead_time")["target"].count()
df['lead_time_rank']=pd.qcut(df['lead_time'].rank(method='first').values,10,duplicates='drop').codes+1
df.groupby("lead_time_rank")["target"].min()
ax = sns.barplot(x = "lead_time_rank", y = "target", data = df, estimator = np.mean)
ax = sns.barplot(x = "arrival_date_year", y = "target", data = df, estimator = np.mean)
df.groupby("arrival_date_year")["target"].mean()
ax = sns.barplot(x = "stays_in_weekend_nights", y = "target", data = df, estimator = np.mean)
df.groupby("stays_in_weekend_nights")["target"].mean()
ax = sns.barplot(x = "stays_in_weekend_nights", y = "target", data = df, estimator = np.mean)
df["stays_in_week_nights_rank"] = pd.qcut(df['stays_in_week_nights'].rank(method='first').values,

                                          5,duplicates='drop').codes+1
ax = sns.barplot(x = "stays_in_week_nights_rank", y = "target", data = df, estimator = np.mean)
df['stay_in_week_night_grp']=np.where(df['stays_in_week_nights_rank'].isin(['1','2']),1,

                                      np.where(df['stays_in_week_nights_rank'].isin(['3','4']),2,3))

df.groupby('stay_in_week_night_grp')['target'].mean()
ax = sns.barplot(x = "is_repeated_guest", y = "target", data = df, estimator = np.mean)
df.groupby("is_repeated_guest")["target"].mean()
ax = sns.barplot(x = "previous_cancellations", y = "target", data = df, estimator = np.mean)
df.groupby("previous_cancellations")["target"].mean()
df["previous_cancellations"].value_counts()
df["prev_cancel_ind"] = np.where(df['previous_cancellations'].isin(["0"]),0,1)

df.groupby("prev_cancel_ind")["target"].count()
ax = sns.barplot(x = "booking_changes", y = "target", data = df, estimator = np.mean)
df.groupby("booking_changes")["target"].mean()
df.groupby("days_in_waiting_list")["target"].mean()
df['day_wait_rank'] = pd.qcut(df['days_in_waiting_list'].rank(method='first').values,5,duplicates='drop').codes+1

ax = sns.barplot(x = "day_wait_rank", y = "target", data=df, estimator = np.mean)
df['day_wait_ind']=np.where(df['day_wait_rank'].isin(['3']),1,0)

df.groupby('day_wait_ind')['target'].count()
df["adr_rank"] = pd.qcut(df["adr"].rank(method='first').values,10 ,duplicates='drop').codes+1

ax = sns.barplot(x = "adr_rank", y = "target", data=df, estimator = np.mean)
ax = sns.barplot(x = "total_of_special_requests", y = "target", data = df, estimator = np.mean)
df.groupby("total_of_special_requests")["target"].mean()
col_num = ["lead_time", "adr"]
col_char = ["total_of_special_requests", "day_wait_ind", "prev_cancel_ind", 

            "stay_in_week_night_grp", "market_segment", "reserved_room_type",

            "distribution_channel"]
x_dummies = pd.get_dummies(df[col_char], drop_first = True)
x_all = pd.concat([df[col_num],x_dummies], axis = 1, join = "inner")
x_var = x_all

y_var = df["target"]
x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, test_size = 0.3, random_state = 42)
lr = LogisticRegression()

lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

score = lr.score(x_test, y_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(score))
confusion_matrix = confusion_matrix(y_test, y_pred_lr)

print(confusion_matrix)
dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)

score = metrics.accuracy_score(y_test,y_pred_dt)

print('Accuracy of Decision Tree on test set: {:.2f}'.format(score))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred_dt)

print(confusion_matrix)
rf = RandomForestClassifier()

rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

score = metrics.accuracy_score(y_test,y_pred_rf)

print('Accuracy of Random Forest Classifier on test set: {:.2f}'.format(score))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred_rf)

print(confusion_matrix)
print(classification_report(y_test, y_pred_rf))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, lr.predict(x_test))

tree_roc_auc = roc_auc_score(y_test, dt.predict(x_test))

RF_roc_auc = roc_auc_score(y_test, rf.predict(x_test))



fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(x_test)[:,1])

fpr, tpr, thresholds = roc_curve(y_test, dt.predict_proba(x_test)[:,1])

fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(x_test)[:,1])



plt.figure()



plt.plot(fpr, tpr, label = 'Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot(fpr, tpr, label = 'Decision Tree (area = %0.2f)' % tree_roc_auc)

plt.plot(fpr, tpr, label = 'Random Foreest Regression (area = %0.2f)' % RF_roc_auc)



plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
y_pred_prob_lr = lr.predict_proba(x_var)[:, 1]

df['y_pred_P_lr'] = pd.DataFrame(y_pred_prob_lr)

df['P_Rank_lr'] = pd.qcut(df['y_pred_P_lr'].rank(method='first').values,10,duplicates='drop').codes+1

df.groupby('P_Rank_lr')['y_pred_P_lr'].sum()
y_pred_prob_dt = dt.predict_proba(x_var)[:, 1]

df['y_pred_P_dt'] = pd.DataFrame(y_pred_prob_dt)

df['P_Rank_dt'] = pd.qcut(df['y_pred_P_dt'].rank(method='first').values,10,duplicates='drop').codes+1

df.groupby('P_Rank_dt')['target'].sum()
y_pred_prob_dt = dt.predict_proba(x_var)[:, 1]

df['y_pred_P_dt'] = pd.DataFrame(y_pred_prob_dt)

df['P_Rank_dt'] = pd.qcut(df['y_pred_P_dt'].rank(method='first').values,10,duplicates='drop').codes+1

df.groupby('P_Rank_dt')['target'].sum()
y_pred_prob_rf = rf.predict_proba(x_var)[:, 1]

df['y_pred_P_rf'] = pd.DataFrame(y_pred_prob_rf)

df['P_Rank_rf']=pd.qcut(df['y_pred_P_rf'].rank(method='first').values,10,duplicates='drop').codes+1

df.groupby('P_Rank_rf')['target'].sum()
df.head()

df.to_csv('hotel_demand_prediction_scored_file.csv')