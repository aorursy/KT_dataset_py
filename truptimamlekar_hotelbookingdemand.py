import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score

from pandas.plotting import scatter_matrix

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc
df=pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

df.head(4)
df.info()
df.describe()
df.isnull().sum()
df= df.fillna(0)
f,ax=plt.subplots(1,1,figsize=(25,8))

sns.countplot(x="hotel",hue="arrival_date_month",data=df,palette="muted")
f,ax=plt.subplots(1,2,figsize=(25,6))

sns.countplot(x="hotel",hue="is_canceled",data=df,ax=ax[0], palette="muted")

sns.countplot(x="hotel",hue="reservation_status",data=df,ax=ax[1], palette="muted")
sns.countplot(x="hotel",data=df, palette="muted")
f,ax=plt.subplots(1,3,figsize=(25,6))

sns.countplot(x="hotel",hue="adults",data=df,ax=ax[0], palette="muted")

sns.countplot(x="hotel",hue="children",data=df,ax=ax[1], palette="muted")

sns.countplot(x="hotel",hue="babies",data=df,ax=ax[2], palette="muted")
f,ax=plt.subplots(1,1,figsize=(15,5))

sns.countplot(x="hotel",hue="meal",data=df, palette="muted")
f,ax=plt.subplots(1,1,figsize=(15,4))

sns.countplot(x="reservation_status",hue="hotel",data=df, palette="muted")
f,ax=plt.subplots(1,1,figsize=(15,4))

sns.countplot(x="deposit_type",hue="hotel",data=df, palette="muted")
f,ax=plt.subplots(1,1,figsize=(15,4))

sns.countplot(x="customer_type",hue="hotel",data=df, palette="muted")
f,ax=plt.subplots(1,1,figsize=(15,4))

sns.countplot(x="market_segment",hue="hotel",data=df, palette="muted")

f,ax=plt.subplots(1,1,figsize=(25,6))

sns.countplot(x="country",data=df.head(500), palette="muted")
f,ax=plt.subplots(1,1,figsize=(25,10))

sns.countplot(x="arrival_date_month",data=df, palette="muted")
f,ax=plt.subplots(2,2,figsize=(25,15))

sns.countplot(x="customer_type",data=df,palette="muted",ax=ax[0][0])

sns.countplot(x="reserved_room_type",data=df,palette="muted",ax=ax[0][1])

sns.countplot(x="assigned_room_type",data=df,palette="muted",ax=ax[1][0])

sns.countplot(x="distribution_channel",data=df,palette="muted",ax=ax[1][1])
f,ax=plt.subplots(1,2,figsize=(25,6))

sns.countplot(x="required_car_parking_spaces",data=df,palette="muted",ax=ax[0])

sns.countplot(x="total_of_special_requests",data=df,palette="muted",ax=ax[1])
sns.set(font_scale=1.5)

plt.figure(figsize=(20,8))

corr = (df.corr())

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values,cmap="YlGnBu",annot=True,linewidths=.5, fmt=".2f")

plt.title("Pearson Correlation of all Elements")
sns.pairplot(df,vars = ['lead_time','arrival_date_year','arrival_date_day_of_month', 'adults', 'children','babies'] )
df_copy = df.copy()

df_copy.head(1)
df_copy.rename(columns={"arrival_date_week_number": "A_Weeknumber", "arrival_date_day_of_month": "A_Datemonth","stays_in_weekend_nights":"weekend_nights","stays_in_week_nights":"week_nights"},inplace=True)

df_copy.head(1)
sns.pairplot(df_copy,hue ='hotel', vars = ['A_Weeknumber','A_Datemonth','lead_time','weekend_nights','week_nights'] )
d1=df.head(100)

df2= d1.groupby('country').size()

df2.plot(kind='pie', subplots=True, figsize=(8, 8))

plt.title("Pie Chart of Various Category")

plt.ylabel("")

plt.show()
f,ax=plt.subplots(1,2,figsize=(25,4))

hist_mean=df['arrival_date_day_of_month'].hist(bins=10, figsize=(15,5),grid=False,ax=ax[0])

hist_mean=df['arrival_date_week_number'].hist(bins=10, figsize=(15,5),grid=False,ax=ax[1])
f,ax=plt.subplots(1,2,figsize=(25,5))

hist_mean=df['agent'].hist(bins=10, figsize=(15,5),grid=False,ax=ax[0])

hist_mean=df['reservation_status_date'].hist(bins=10, figsize=(15,4),grid=False,ax=ax[1])
f,ax=plt.subplots(1,2,figsize=(25,6))

sns.violinplot(x="hotel", y="lead_time",ax=ax[0],data=df, palette="muted")

sns.violinplot(x="hotel", y="arrival_date_week_number",data=df,ax=ax[1], palette="muted")
df['arrival_date_month'].replace([0], 'July', inplace=True) 

df['arrival_date_month'].replace([1], 'August', inplace=True) 

df['arrival_date_month'].replace([2], 'May', inplace=True)   

f,ax=plt.subplots(1,1,figsize=(25,10))

sns.kdeplot(df.loc[(df['arrival_date_month']=='July'), 'lead_time'], color='b', shade=True, Label='July')

sns.kdeplot(df.loc[(df['arrival_date_month']=='August'), 'lead_time'], color='g', shade=True, Label='August')

sns.kdeplot(df.loc[(df['arrival_date_month']=='May'), 'lead_time'], color='r', shade=True, Label='May')

plt.xlabel('Lead_Time') 

plt.ylabel('Probability Density')
f,ax=plt.subplots(1,3,figsize=(25,8))

sns.scatterplot(x="arrival_date_week_number", y="hotel",color = "red",data=df,ax=ax[0])

sns.scatterplot(x="arrival_date_day_of_month", y="hotel",color = "green",data=df,ax=ax[1])

sns.scatterplot(x="lead_time", y="hotel",color = "orange",data=df,ax=ax[2])
f,ax=plt.subplots(1,3,figsize=(25,8))

df['lead_time'].plot.box(sym='.k',figsize=(25,7),ax=ax[0])

df['arrival_date_week_number'].plot.box(sym='.k',figsize=(25,7),ax=ax[1])

df['arrival_date_day_of_month'].plot.box(sym='.k',figsize=(25,7),ax=ax[2])

plt.show ()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['hotel']=le.fit_transform(df['hotel'])

df['arrival_date_month']=le.fit_transform(df['arrival_date_month'])

df['meal'] = le.fit_transform(df['meal'])

df['market_segment'] = le.fit_transform(df['market_segment'])

df['distribution_channel'] = le.fit_transform(df['distribution_channel'])

df['reserved_room_type'] = le.fit_transform(df['reserved_room_type'])

df['assigned_room_type'] =le.fit_transform(df['assigned_room_type'])

df['customer_type'] = le.fit_transform(df['customer_type'])

df['reservation_status'] = le.fit_transform(df['reservation_status'])

df['deposit_type'] = le.fit_transform(df['deposit_type'])
from sklearn.model_selection import train_test_split

X = df.drop(['is_canceled','reservation_status_date','country'], axis=1)

Y = df['is_canceled']

x_train, x_test, y_train, y_test = train_test_split(X,Y,random_state=0)
logistic = LogisticRegression()

logistic.fit(x_train,y_train)

y_pred=logistic.predict(x_test)

print(classification_report(y_test,y_pred))

accuracy1=logistic.score(x_test,y_test)

print (accuracy1*100,'%')

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True)
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize = (10,6))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, color = 'red', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1], linestyle = '--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
train_score = logistic.score(x_train,y_train)

test_score = logistic.score(x_test,y_test)

print(f'Training Accuracy of our model is: {train_score}')

print(f'Test Accuracy of our model is: {test_score}')
ran_class=RandomForestClassifier()

ran_class.fit(x_train,y_train)

ran_predict=ran_class.predict(x_test)

print(classification_report(y_test,ran_predict))

accuracy3=ran_class.score(x_test,y_test)

print(accuracy3*100,'%')

cm = confusion_matrix(y_test, ran_predict)

sns.heatmap(cm, annot= True)
train_score = ran_class.score(x_train,y_train)

test_score = ran_class.score(x_test,y_test)

print(f'Training Accuracy of our model is: {train_score}')

print(f'Test Accuracy of our model is: {test_score}')
prediction = ran_class.predict(x_train.iloc[15].values.reshape(1,-1))

actual_value = y_train.iloc[15]

print(f'Predicted Value \t: {prediction[0]}')

print(f'Actual Value\t\t: {actual_value}')