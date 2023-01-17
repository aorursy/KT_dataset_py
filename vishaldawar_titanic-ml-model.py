# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# ML libraries



from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score, accuracy_score

import category_encoders as ce

from sklearn.model_selection import GridSearchCV



# Visualisation libraries

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')





train.shape, test.shape
train.head(3)
train.info()
f, ax = plt.subplots(ncols=2,figsize=(18,6))



sns.distplot(train[train['Survived']==0]['Age'],ax=ax[0],label='Not Survived')

sns.distplot(train[train['Survived']==1]['Age'],ax=ax[0],label='Survived')

ax[0].legend()

sns.boxplot(x='Survived',y='Age',data=train,ax=ax[1])

ax[1].legend()
train['Age'].fillna(train['Age'].median(),inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)
train.info()
test.info()
test['Age'].fillna(train['Age'].median(),inplace=True)

test['Fare'].fillna(train['Fare'].median(),inplace=True)
fig, ax = plt.subplots(ncols=2, figsize=(18,6))

sns.boxplot(data = train, x = 'Survived', y = 'Fare', showfliers=False, ax=ax[0])

sns.boxplot(data = train, x = 'Survived', y = 'Age', showfliers=False, ax=ax[1])
train.describe()
fare_bins = ['0-10','11-30','31-50','51-100','101-250','251+']

age_bins = ['0-5','6-10','11-17','17-25','26-30','31-35','36-40','41-45','46-50','51-55','56-65','70+']



def fare_bins_func(row, fare_bins):

    for bin in fare_bins[:-1]:

        ind = bin.index('-')

        if row['Fare'] >= int(bin[:ind]) and row['Fare'] <= int(bin[ind+1:]):

            flag = 1

            val = bin

            break

        else:

            flag = 0

            val = fare_bins[-1]

    return val

        

def age_bins_func(row, age_bins):

    for bin in age_bins[:-1]:

        ind = bin.index('-')

        if row['Age'] >= int(bin[:ind]) and row['Age'] <= int(bin[ind+1:]):

            val = bin

            break

        else:

            val = age_bins[-1]

    return val
dataset = [train, test]



for df in dataset:

    df['fare_bin'] = df.apply(fare_bins_func, args = [fare_bins],axis = 1)

    df['age_bin'] = df.apply(age_bins_func, args = [age_bins],axis = 1)
fig,ax = plt.subplots(ncols=2,figsize=(18,6))

sns.countplot(x='fare_bin',data=train,ax=ax[0],hue='Survived',order=fare_bins)

sns.countplot(x='age_bin',data=train,ax=ax[1],hue='Survived',order=age_bins)
train['fare_plus_age'] = train['Fare'] + train['Age']

test['fare_plus_age'] = test['Fare'] + test['Age']

train['fare_into_age'] = train['Fare']*train['Age']

test['fare_into_age'] = test['Fare']*test['Age']
le = LabelEncoder()

train['fare_bin_label'] = le.fit_transform(train['fare_bin'])

test['fare_bin_label'] = le.transform(test['fare_bin'])

train['age_bin_label'] = le.fit_transform(train['age_bin'])

test['age_bin_label'] = le.transform(test['age_bin'])
l_train = [name.split(',')[1] for name in train['Name'].values.tolist()]

l_test = [name.split(',')[1] for name in test['Name'].values.tolist()]
titles_train = []

titles_test = []

for val in l_train:

    ind = val.index('.')

    titles_train.append(val[1:ind])

    

for val in l_test:

    ind = val.index('.')

    titles_test.append(val[1:ind])
np.unique(titles_train), np.unique(titles_test)
train['title'] = titles_train

test['title'] = titles_test
test['title'] = np.where(test['title'] == 'Dona','Don',test['title'])
np.unique(train['title']), np.unique(test['title'])
sex_enc = LabelEncoder()

sex_enc.fit(train['Sex'])

sex_enc.classes_
sex_labels_train = sex_enc.transform(train['Sex'].values)

sex_labels_test = sex_enc.transform(test['Sex'].values)

train['sex'] = sex_labels_train

test['sex'] = sex_labels_test
def sex_category(row):

    if row['sex'] == 0 :

        if row['title'] != 'Miss' and row['Age'] >= 18:

            return 'mrs'

        elif row['title'] == 'Miss':

            return 'miss'

        else:

            return 'msc'

    elif row['title'] == 'Master':

        return 'mtr'

    elif row['title'] == 'Mr':

        return 'mr'

    else:

        return 'msc'
dataset = [train, test]
for df in dataset:

    df['encoded_title'] = df.apply(sex_category, axis=1)

    
np.unique(train['encoded_title']), np.unique(test['encoded_title'])
for df in dataset:

    df['title_sex'] = df['encoded_title'] + '_' + df['sex'].astype('str')
train['Embarked'].value_counts()
train.head(3)
le = LabelEncoder()

le.fit(train['title_sex'])

for df in dataset:

    df['title_sex_label'] = le.transform(df['title_sex'])
np.unique(train['title_sex']),np.unique(test['title_sex'])
sns.countplot(x='title_sex',data=train,hue='Survived')
le = LabelEncoder()

le.fit(train['Embarked'])



train['embarked'] = le.transform(train['Embarked'])

test['embarked'] = le.transform(test['Embarked'])
encoding = train.groupby('Embarked').size()

# get frequency of each category

encoding = encoding/len(train)

train['enc'] = train.Embarked.map(encoding)



encoding = test.groupby('Embarked').size()

# get frequency of each category

encoding = encoding/len(test)

test['enc'] = test.Embarked.map(encoding)
train['family_size'] = train['SibSp'] + train['Parch'] + 1

test['family_size'] = test['SibSp'] + test['Parch'] + 1
train['isAlone'] = np.where(train['family_size'] == 1, 0, 1)

test['isAlone'] = np.where(test['family_size'] == 1, 0, 1)
train.info()
fig, ax = plt.subplots(ncols=3, figsize=(18,6))

sns.countplot(x='embarked',data=train,hue='Survived',ax=ax[0])

sns.countplot(x='Pclass',data=train,hue='Survived',ax=ax[1])

sns.countplot(x='title_sex_label',data=train,hue='Survived',ax=ax[2])
columns = list(train.describe().columns)

columns.remove('PassengerId')

columns_test = columns

print(columns)

train_data = train[columns]

columns_test.remove('Survived')

train_X_data = train_data[columns_test]

train_y_data = train_data[['Survived']]

test_X = test[columns_test]
train_X_data.isna().sum(), test_X.isna().sum()
train_X_data.describe()
train_X,validation_X,train_y,validation_y = train_test_split(train_X_data,train_y_data,test_size=0.25)
train_X.shape, validation_X.shape,train_y.shape,validation_y.shape
rf = RandomForestClassifier(n_estimators=200,max_depth=20,max_features=10,n_jobs=-1,random_state=42)
rf.fit(train_X,train_y)
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(rf, random_state=1).fit(validation_X, validation_y)

eli5.show_weights(perm, feature_names = validation_X.columns.tolist())
cols_removed = ['age_bin_label','fare_bin_label']

train_X = train_X.drop(cols_removed,1)

validation_X = validation_X.drop(cols_removed,1)

test_X = test_X.drop(cols_removed,1)
xgb = XGBClassifier(n_estimators=1000,early_stop_round=10,

                   eval_set=[validation_X,validation_y], learning_rate=0.05, 

                    max_depth=10,n_jobs=-1,random_state=42)

rf = RandomForestClassifier(n_estimators=1000,max_depth=20,max_features=10,n_jobs=-1,random_state=42)
xgb.fit(train_X,train_y)

rf.fit(train_X,train_y)
train_X.shape
xgb_train_proba = xgb.predict_proba(train_X)[:,1]

rf_train_proba = rf.predict_proba(train_X)[:,1]



xgb_val_proba = xgb.predict_proba(validation_X)[:,1]

rf_val_proba = rf.predict_proba(validation_X)[:,1]
train_prec_xgb, train_rec_xgb, train_thresholds_xgb_1 = precision_recall_curve(train_y, xgb_train_proba)

train_fpr_xgb, train_tpr_xgb, train_thresholds_xgb_2 = roc_curve(train_y, xgb_train_proba)

train_prec_rf, train_rec_rf, train_thresholds_rf_1 = precision_recall_curve(train_y, rf_train_proba)

train_fpr_rf, train_tpr_rf, train_thresholds_rf_2 = roc_curve(train_y, rf_train_proba)
prec_xgb, rec_xgb, thresholds_xgb_1 = precision_recall_curve(validation_y, xgb_val_proba)

fpr_xgb, tpr_xgb, thresholds_xgb_2 = roc_curve(validation_y, xgb_val_proba)

prec_rf, rec_rf, thresholds_rf_1 = precision_recall_curve(validation_y, rf_val_proba)

fpr_rf, tpr_rf, thresholds_rf_2 = roc_curve(validation_y, rf_val_proba)
f, ax = plt.subplots(2,2,figsize=(18,10))

ax[0][0].plot(train_fpr_xgb, train_tpr_xgb, color='orange', label='xgb')

ax[0][0].plot(train_fpr_rf, train_tpr_rf, color='darkblue', label='rf')

ax[0][0].plot([0, 1], [0, 1], color='darkblue', linestyle='--')

ax[0][0].set_xlabel('FPR')

ax[0][0].set_ylabel('TPR')

ax[0][0].set_title('ROC value')

ax[0][0].legend()

ax[0][1].plot(train_rec_xgb, train_prec_xgb,color='orange',label='xgb')

ax[0][1].plot(train_rec_rf, train_prec_rf,color='darkblue',label='rf')

ax[0][1].set_xlabel('Recall')

ax[0][1].set_ylabel('Precision')

ax[0][1].set_title('Recall vs Precision')

ax[0][1].legend()

ax[1][0].plot(train_thresholds_xgb_1,train_prec_xgb[:-1],color='orange',label='xgb')

ax[1][0].plot(train_thresholds_rf_1,train_prec_rf[:-1],color='darkblue',label='rf')

ax[1][0].set_xlabel('Threshold')

ax[1][0].set_ylabel('Precision')

ax[1][0].set_title('Threshold vs Precision')

ax[1][0].legend()

ax[1][1].plot(train_thresholds_xgb_1,train_rec_xgb[:-1],color='orange',label='xgb')

ax[1][1].plot(train_thresholds_rf_1,train_rec_rf[:-1],color='darkblue',label='rf')

ax[1][1].set_xlabel("Thresholds")

ax[1][1].set_ylabel("Recall")

ax[1][1].set_title("Threshold vs Recall")

ax[1][1].legend()

plt.show()
f, ax = plt.subplots(2,2,figsize=(18,10))

ax[0][0].plot(fpr_xgb, tpr_xgb, color='orange', label='xgb')

ax[0][0].plot(fpr_rf, tpr_rf, color='darkblue', label='rf')

ax[0][0].plot([0, 1], [0, 1], color='darkblue', linestyle='--')

ax[0][0].set_xlabel('FPR')

ax[0][0].set_ylabel('TPR')

ax[0][0].set_title('ROC value')

ax[0][0].legend()

ax[0][1].plot(rec_xgb,prec_xgb,color='orange',label='xgb')

ax[0][1].plot(rec_rf,prec_rf,color='darkblue',label='rf')

ax[0][1].set_xlabel('Recall')

ax[0][1].set_ylabel('Precision')

ax[0][1].set_title('Recall vs Precision')

ax[0][1].legend()

ax[1][0].plot(thresholds_xgb_1,prec_xgb[:-1],color='orange',label='xgb')

ax[1][0].plot(thresholds_rf_1,prec_rf[:-1],color='darkblue',label='rf')

ax[1][0].set_xlabel('Threshold')

ax[1][0].set_ylabel('Precision')

ax[1][0].set_title('Threshold vs Precision')

ax[1][0].legend()

ax[1][1].plot(thresholds_xgb_1,rec_xgb[:-1],color='orange',label='xgb')

ax[1][1].plot(thresholds_rf_1,rec_rf[:-1],color='darkblue',label='rf')

ax[1][1].set_xlabel("Thresholds")

ax[1][1].set_ylabel("Recall")

ax[1][1].set_title("Threshold vs Recall")

ax[1][1].legend()

plt.show()
def class_predict(prob, thresh):

    a = np.where(prob >= thresh, 1, 0)

    return a
test_X = test[train_X.columns.tolist()]
train_proba = rf.predict_proba(train_X)[:,1]

val_proba = rf.predict_proba(validation_X)[:,1]

test_proba = rf.predict_proba(test_X)[:,1]
thresh = 0.6



train_pred = class_predict(train_proba, thresh)

val_pred = class_predict(val_proba, thresh)

test_pred = class_predict(test_proba, thresh)



print("Precision : ",round(precision_score(train_y, train_pred),3),

      " Recall : ",round(recall_score(train_y, train_pred),3),

      " Accuracy : ",round(accuracy_score(train_y,train_pred),3),

     "AUC-ROC : ",round(roc_auc_score(train_y,train_proba),3))



print("Precision : ",round(precision_score(validation_y, val_pred),3),

      " Recall : ",round(recall_score(validation_y, val_pred),3),

      " Accuracy : ",round(accuracy_score(validation_y,val_pred),3),

     "AUC-ROC : ",round(roc_auc_score(validation_y,val_proba),3))



thresholds = np.linspace(0,1,1000)

prec = []

rec = []

acc = []

for thresh in thresholds:

    pred = class_predict(val_proba,thresh)

    p = precision_score(validation_y,pred)

    r = recall_score(validation_y,pred)

    a = accuracy_score(validation_y,pred)

    prec.append(p)

    rec.append(r)

    acc.append(a)
fig, ax = plt.subplots(ncols=3,figsize=(18,6))



ax[0].plot(thresholds,prec)

ax[1].plot(thresholds,rec)

ax[2].plot(thresholds,acc)

ax[0].set_xlabel('thresh')

ax[1].set_xlabel('thresh')

ax[2].set_xlabel('thresh')

ax[0].set_ylabel('precision')

ax[1].set_ylabel('recall')

ax[2].set_ylabel('accuracy')

ax[0].set_title('Threshold vs precision')

ax[1].set_title('Threshold vs recall')

ax[2].set_title('Threshold vs accuracy')

plt.show()
ind = acc.index(max(acc))

print(thresholds[ind],prec[ind],rec[ind],acc[ind])
test_pred = class_predict(test_proba, thresholds[ind])
submission = test[['PassengerId']]

submission['Survived'] = test_pred
submission.to_csv('/kaggle/working/submssion.csv',index=False)