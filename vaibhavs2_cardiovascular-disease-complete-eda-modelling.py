# Importing important libraries.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
plt.style.use('seaborn')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/cardiovascular-disease-dataset/cardio_train.csv',sep=';')
print(f'Dataset Shape: {data.shape}')
# Identifying missing values and duplicates first.
data.isna().sum()
duplicates = len(data) - len(data.drop(['id'],axis=1).drop_duplicates())
data.drop(['id'],axis=1,inplace=True)
data.drop_duplicates(inplace=True)
print(f'{duplicates} duplicate records dropped.')
data.shape
# Let us now begin first with finding some quick descriptive stats about our data.
print(f'{data.dtypes.value_counts()}')
print('Let us now get a quick summary of features available.')
data.describe().T.round(2)
# Let us first have a look at our target variable.
fig, ax = plt.subplots(1,1)
sns.countplot(data['cardio'], ax = ax)
for i in ax.patches:
    height = i.get_height()
    ax.text(i.get_x()+i.get_width()/2,height,'{:.2f}'.format((i.get_height()/len(data['cardio']))*100,'%'))
plt.show()
# Age is given in days. Transforming it into years for better understanding and checking relation with the target variable.
data['age'] = data['age']/365
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
sns.distplot(data['age'][data['cardio']==0], ax = ax1, color='green')
sns.distplot(data['age'][data['cardio']==1], ax = ax1,color='coral')
ax1.set_title('Age Distribution')
ax1.legend()

sns.distplot(data['age'][(data['gender']==1) & (data['cardio']==1)],ax = ax2,color='pink')
sns.distplot(data['age'][(data['gender']==2) & (data['cardio']==1)],ax = ax2,color='blue')
ax2.set_title('Disease count distribution by  gender, aged below 54.')
plt.show()
fig, (ax1) = plt.subplots(1,1, figsize=(10,10))
sns.boxenplot(data['cardio'],(data['height']*0.0328084),ax=ax1)
ax1.set_title('Height / Diseased')
plt.show()
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
sns.scatterplot(data['age'],data['height'][(data['height']*0.0328084)<4]*0.0328084,hue=data['cardio'],ax=ax1)
ax1.set_title('Height vs Age')
sns.scatterplot(data['weight'],data['height'][(data['height']*0.0328084)<4]*0.0328084,hue=data['cardio'],ax=ax2)
ax2.set_title('Height vs Weight')
plt.show()
# Converting height in cms to foot.
data['height'] = data['height']*0.0328084 
filt =(data['height']>8) | (data['height']<3) 

data.drop(index = list(data[filt].index),inplace=True)
print(f'Dataset: {data.shape}')
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,5))
sns.boxenplot(data['cardio'],(data['weight']),ax=ax1)
ax1.set_title('Weight / Diseased')
sns.scatterplot(data['weight'],data['height'],ax=ax2,hue=data['cardio'])
ax2.set_title('height vs weight')
plt.show()
# 1. Weight < 25 kg
filt1 = data['weight']<25
data.drop(index=list(data[filt1].index),inplace=True)

# 2. Weight > 175 kg
filt2 = data['weight']>175
data.drop(index=list(data[filt2].index),inplace=True)

# 3. Height < 4.5 & Weight > 150 kg
filt3 = (data['height']<4.5) & (data['weight']>150)
data.drop(index=list(data[filt3].index),inplace=True)
# Gender
fig,(ax) = plt.subplots(1,1)
tmp = pd.crosstab(data['gender'],data['cardio'],normalize='index').round(4)*100
tmp.reset_index()
tmp.columns = ['Not Diseased','Diseased']
ax1 = sns.countplot(data['gender'],order = list(tmp.index))
ax2 = ax1.twinx()
sns.pointplot(tmp.index,tmp['Diseased'],order = list(tmp.index),ax=ax2, color='red')
for x in ax1.patches:
    height = x.get_height()
    ax1.text(x.get_x()+x.get_width()/2,height,'{:.2f}{}'.format((height/len(data))*100,'%'))
plt.show()
# ap_hi
filt = (data['ap_hi']<90) | (data['ap_hi']>140)
print(f'Normal systolic blood pressure range is between 90 and 120. However, from our dataset we can see that we have {len(data[filt])} records that are not falling within the normal range. We can replace them with their median values.')
data['ap_hi'].replace(data[filt]['ap_hi'].values,data['ap_hi'].median(),inplace=True)
# filt =  (data['ap_lo']>90) | (data['ap_lo']<60)
fig, ax = plt.subplots(1,1, figsize = (30,10))
sns.distplot(data['ap_lo'][data['ap_lo']<200],bins = 25, kde = True, ax = ax)
xticks = [i*10 for i in range(-5,20)]
ax.set_xticks(xticks)
ax.tick_params(x,labelrotation='v')
plt.show()
print(f'Similar to Systolic Blood Pressure Range the diastolic bp range should be between 60-90 for a healthy individual. However, in this case we have median values for AP_LO as {data.ap_lo.median()} which does not look correct to me. Considering this in mind we would have to do some further analysis if the data source is correct or not.')
# data.replace(data[filt]['ap_lo'].values,data['ap_lo'].median(),inplace=True)
sns.boxenplot(data['cardio'],data['ap_lo'][data['ap_lo']<150])
plt.show()
# cholesterol
tmp = pd.crosstab(data['cholesterol'],data['cardio'],normalize='index')
tmp.reset_index()
tmp.columns = ['not diseased','diseased']
fig, ax = plt.subplots(1,1)
sns.countplot(data['cholesterol'],order=list(tmp.index), ax=ax)
plot2 = ax.twinx()
sns.pointplot(tmp.index,tmp['diseased'],order=list(tmp.index),ax=plot2)
for patch in ax.patches:
    height = patch.get_height()
    ax.text(patch.get_x()+patch.get_width()/2,height,'{:.2f}{}'.format(height/len(data['cholesterol'])*100,'%'))
plt.show()


# Glucose
tmp = pd.crosstab(data['gluc'],data['cardio'],normalize='index')
tmp.reset_index()
tmp.columns = ['not diseased','diseased']
fig, ax = plt.subplots(1,1)
sns.countplot(data['gluc'],order=list(tmp.index), ax=ax)
plot2 = ax.twinx()
sns.pointplot(tmp.index,tmp['diseased'],order=list(tmp.index),ax=plot2)
for patch in ax.patches:
    height = patch.get_height()
    ax.text(patch.get_x()+patch.get_width()/2,height,'{:.2f}{}'.format(height/len(data['gluc'])*100,'%'))
plt.show()
data['smoke/drink'] = data['smoke'].apply(str)+'|'+data['alco'].apply(str)

tmp = pd.crosstab(data['smoke/drink'],data['cardio'],normalize='index')
tmp.reset_index()
tmp.columns = ['Not diseased','diseased']

fig, ax = plt.subplots(1,1)
sns.countplot(data['smoke/drink'],order=list(tmp.index), ax=ax)
plot2 = ax.twinx()
sns.pointplot(tmp.index,tmp['diseased'],order=list(tmp.index),ax=plot2)
for patch in ax.patches:
    height = patch.get_height()
    ax.text(patch.get_x()+patch.get_width()/2,height,'{:.2f}{}'.format(height/len(data['smoke/drink'])*100,'%'))
plt.show()
df_smoke_drink = pd.get_dummies(data['smoke/drink'],prefix='smoke/drink',drop_first=True)
data = pd.concat([data,df_smoke_drink],axis=1)
data.drop(['smoke/drink'],axis=1,inplace=True)
# data.head()
# BMI = weight(kg)/height(m2)
data['BMI'] = data['weight']/(((data['height']/0.0328084)*.01)**2)
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
sns.boxenplot(data['cardio'],data['BMI'],ax=ax1)
sns.distplot(data[data['cardio']==0]['BMI'],color='g',ax=ax2)
sns.distplot(data[data['cardio']==1]['BMI'],color='b',ax=ax2)
plt.show()
# The very first thing that we need to do is to break our data into training and test sets. 
from sklearn.model_selection import train_test_split
train,test = train_test_split(data, test_size = 0.25, random_state=42)
print (f'The shapes of our train & test data is {train.shape} and {test.shape} respectively.')
# Logistic Regression model assumes that there should be no multi-colinearity amongst the variables. 
fig, ax = plt.subplots(1,1, figsize=(20,10))
sns.heatmap(train.corr().sort_values(by='cardio'), annot=True)
plt.show()
# Logistic Regresssion - Selecting best penalty value for our Regularized model in scikit- learn

X = np.array(train.drop(['cardio','height','weight','gender','alco','smoke'], axis=1))
y = np.array(train['cardio'])
X_test = np.array(test.drop(['cardio','height','weight','gender','alco','smoke'], axis=1))
y_act = np.array(test['cardio'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state=42)

from sklearn.linear_model import LogisticRegression
log_classifier = LogisticRegression()

from sklearn.model_selection import GridSearchCV
params = {'C':[0.001, 0.1,1,10,100,1000]}
grid = GridSearchCV(log_classifier, cv=kfold, param_grid=params)
grid.fit(X,y)
grid.best_params_
from sklearn.metrics import accuracy_score, confusion_matrix
log_classifier = LogisticRegression(C=10)

log_classifier.fit(X,y)
print(f'Train Score: {log_classifier.score(X,y)}')

y_pred = log_classifier.predict(X_test)
print(f'Test Score: {accuracy_score(y_act,y_pred)}')
# Statsmodel implementation.

import statsmodels.api as sm
x1 = sm.add_constant(X)
features = list(train.drop(['cardio','height','weight','gender','alco','smoke'],axis=1).columns)
features.insert(0,'const')
log_reg = sm.Logit(y,x1)
results = log_reg.fit()
results.summary(xname=features)
print(f'Accuracy {(results.pred_table()[0][0]+results.pred_table()[1][1])/len(train)}')
print(f'{results.pred_table()}')
X_test = test.drop(['cardio','height','weight','gender','alco','smoke'], axis=1)
x1_test = sm.add_constant(X_test)
y_pred = results.predict(x1_test)
# Decision Tree Classifier

X = np.array(train.drop(['cardio'], axis=1))
y = np.array(train['cardio'])
X_test = np.array(test.drop(['cardio'], axis=1))
y_act = np.array(test['cardio'])

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'gini', random_state=42, max_depth=10)
dt.fit(X,y)
print(f'Train Accuracy for Decision Tree is : {dt.score(X,y)}')
print(f'Test Accuracy Score for Decision Tree is ; {accuracy_score(y_act,dt.predict(X_test))}')
confusion_matrix(y_act,dt.predict(X_test))
# Support Vector Machines
X = np.array(train.drop(['cardio'],axis=1))
y = np.array(train['cardio'])
X_test = np.array(test.drop(['cardio'],axis=1))
y_act = test['cardio']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC
svc = SVC()

# params = {'C':[0.1, 1, 10, 100], 'gamma':[1, 0.1, 0.01, 0.001]}
# grid = GridSearchCV(svc, param_grid=params)
# grid.fit(X,y)
# grid.best_params_

svc.fit(X,y)
print(f'Train Score: {svc.score(X,y)}')


y_pred = svc.predict(X_test)
print(f'Test Score: {accuracy_score(y_act,y_pred)}')
# Naive Bayes Classifier
X = np.array(train.drop(['cardio'],axis=1))
y = np.array(train['cardio'])
X_test = np.array(test.drop(['cardio'],axis=1))
y_act = test['cardio']

from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()
gb.fit(X,y)
print(f'Train Score: {gb.score(X,y)}')


y_pred = gb.predict(X_test)
print(f'Test Score: {accuracy_score(y_act,y_pred)}')


# Random Forest

X = np.array(train.drop(['cardio'], axis=1))
y = np.array(train['cardio'])
X_test = np.array(test.drop(['cardio'], axis=1))
y_act = np.array(test['cardio'])

from sklearn.ensemble import RandomForestClassifier
param = {'n_estimators': [10, 20, 40, 80, 160, 300], 'random_state': [42], 'criterion': ['gini'], 'max_depth': [2, 4, 8, 16, 32]}
rf = RandomForestClassifier()
grid = GridSearchCV(rf,param)
grid.fit(X,y)
grid.best_params_

rf_upd = RandomForestClassifier(n_estimators=40, criterion='gini', max_depth=8, random_state=42)
rf_upd.fit(X,y)
print(f'Train Score: {rf_upd.score(X,y)}')

y_pred = rf_upd.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_act,y_pred)}')
Feature_importances = pd.concat([pd.Series(test.drop(['cardio','ap_lo'], axis=1).columns),pd.Series(rf_upd.feature_importances_)],axis=1).sort_values(by=1, ascending=False)
Feature_importances.columns = ['Feature','Weights']
Feature_importances
# Gradient Boosting 
X = np.array(train.drop(['cardio'],axis=1))
y = np.array(train['cardio'])
X_test = np.array(test.drop(['cardio'],axis=1))
y_act = test['cardio']

from  xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X,y)
print(f'Train Score: {xgb.score(X,y)}')

y_pred = xgb.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_act,y_pred)}')
# ANN
import tensorflow as tf
import keras as ks
from keras.models import Sequential
from keras.layers import Dense

X = np.array(train.drop(['cardio'],axis=1))
y = np.array(train['cardio'])
X_test = np.array(test.drop(['cardio'],axis=1))
y_act = test['cardio']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.fit_transform(X_test)

classifier = Sequential()


classifier.add(Dense(8, input_shape=(15,), activation = 'relu'))
classifier.add(Dense(8, activation = 'relu'))

classifier.add(Dense(1, activation = 'sigmoid'))

classifier.compile('adam',loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X,y, batch_size= 10, epochs = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred >= 0.5)
print(f'Test Accuracy for Deep Learning: {accuracy_score(y_act,y_pred)}')
