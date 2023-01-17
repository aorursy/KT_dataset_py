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

import matplotlib.pyplot as plt

import warnings

import seaborn as sns

warnings.filterwarnings('ignore')

plt.style.use('ggplot')
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
train.columns = train.columns.str.lower()
train.head()
train.shape
train.isnull().sum() / train.shape[0] * 100
import missingno as msno

msno.bar(train,color = (0.9,0.4,1))
corr_matrix = train.corr()

sns.heatmap(corr_matrix,annot=True)
# survived ratio

train['survived'].value_counts().plot(kind='pie',autopct='%.2f%%')

plt.title('survived rate')
train2 = train.copy()

train2['die'] = 1-train['survived']

fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(12,8))

train2.groupby('sex').agg('sum')[['survived','die']].plot(kind='bar',ax=ax1)

ax1.set(title = 'survived rate by sex')

train2.groupby('pclass').agg('sum')[['survived','die']].plot(kind='bar',ax=ax2)

ax2.set(title = 'survived rate by pclass')
y_position = 1.02

fig, axes = plt.subplots(1,2 , figsize = (18,8))

train['pclass'].value_counts().plot.bar(color = ['#CD7F32', '#FFDF00', '#D3D3D3'] , ax = axes[0] )

axes[0].set_title('Number of passengers by Pclass', y = y_position)

axes[0].set_ylabel('Count')

sns.countplot('pclass', hue = 'survived', data = train, ax = axes[1])

axes[1].set_title('Pclass: Survived vs Dead' , y = y_position)

plt.show()
fig, axes = plt.subplots(1,2, figsize = (18,8))

train[['sex','survived']].groupby(['sex'], as_index = True).mean().plot.bar(ax = axes[0])

axes[0].set_title('Survived vs Sex') 

sns.countplot('sex' , hue = 'survived', data = train, ax = axes[1])

axes[1].set_title('Sex: Survived vs Dead')

plt.show 
train['embarked'].fillna(train['embarked'].mode()[0],inplace=True) # Replace missing values with mode.

# Through name , we can predict job and age

train['job'] = train['name'].apply(lambda x: x.split(',')[1].split('.')[0].lstrip())
train['fare'] = train['fare'].apply(np.round)
train.head()
train['family_size']= train['sibsp']+train['parch']+1

# 1 - alone, 2~4 - small , 5~ big

train['family_size'] = pd.cut(train['family_size'],[0,2,4,12],labels = ['alone','small','big'],right=False)
train['title'] = train['job'].copy()

train['title'] = train['title'].apply(lambda x: x.lower())
# key: old, value:new

title_map={ 'capt':'officer',

           'col':'officer'

           ,'major':'officer'

           ,'dr':'officer'

           ,'rev':'officer'

           ,'sir':'officer',

           'jonkheer': 'royalty',

          'the countess':'royalty',

          'dona': 'royalty',

          'lady': 'royalty',

          'don': 'royalty',

          'mr': 'mr',

          'mrs':'mrs',

          'ms':'mrs',

          'mme':'mrs',

          'miss':'miss',

          'mlle':'miss',

          'master':'master'}
train['title'] = train['title'].map(title_map)
sns.boxplot(data=train,x='title',y='age')
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 400)

tmp = train[['age','title']]
from sklearn.preprocessing import OneHotEncoder

def ohe_trans(data,col):

    ohe=OneHotEncoder()

    x= ohe.fit_transform(data[col].values.reshape(-1,1)).toarray()

    tp = []

    for i in range(data[col].unique().size):

        tp.append(col[0]+str(i))

    ohe_df = pd.DataFrame(x,columns = tp)

    return ohe_df

tmp = pd.concat([tmp,ohe_trans(tmp,'title')],axis=1)

tmp.drop('title',axis=1,inplace=True)
X = tmp[tmp['age'].notnull()]

y = tmp[tmp['age'].isnull()]

train_x = X.loc[:,X.columns.difference(['age'])]

train_y = X['age']

test_x = y.loc[:,y.columns.difference(['age'])]

test_y = y['age']
rf.fit(train_x,train_y)
pred = rf.predict(test_x)

train['age'][y.index] = pred
def alone(x):

    if x == 'alone':

        return 1

    else:

        return 0

train['isalone'] = train['family_size'].apply(lambda x: alone(x))
train['age'] = pd.cut(train['age'],[0,20,40,60,100],labels = [0,1,2,3],right=False)

train['cabin'] = train['cabin'].apply(lambda x: 0 if type(x) == float else 1)
size_map={'alone':0,'small':1,'big':2}

train['family_size'] = train['family_size'].map(size_map)
train.drop(['sibsp','parch','ticket','name','job','passengerid'],axis=1,inplace=True)
test.head()
passengerid = test['PassengerId']
test.isnull().sum() / test.shape[0] * 100
test.columns = test.columns.str.lower()
# Add to family_size column

test['family_size'] = test['sibsp'] + test['parch'] + 1

pd.cut(test['family_size'],[0,2,4,12],labels = ['alone','small','big'],right=False)

# Add to title column

test['title'] = test['name'].apply(lambda x: x.split(',')[1].split('.')[0].lstrip())

test['title'] = test['title'].apply(lambda x: x.lower())

test['title'] = test['title'].map(title_map)

# Add to isalone colum

test['isalone'] = test['family_size'].apply(lambda x: alone(x))

# family_size column tunning

test['family_size'].map(size_map)

# embarked column tunning

test['cabin'] = test['cabin'].apply(lambda x: 0 if type(x) == float else 1)

# Age column missing value processing

tmp = test[['age','title']]

tmp = pd.concat([tmp,ohe_trans(tmp,'title')],axis=1)

tmp.drop('title',axis=1,inplace=True)

X = tmp[tmp['age'].notnull()]

y = tmp[tmp['age'].isnull()]

train_x = X.loc[:,X.columns.difference(['age'])]

train_y = X['age']

test_x = y.loc[:,y.columns.difference(['age'])]

test_y = y['age']

rf.fit(train_x,train_y)

pred = rf.predict(test_x)

test['age'][y.index]=pred

test['age'] = pd.cut(test['age'],[0,20,40,60,100],labels = [0,1,2,3],right=False)
tmp=[]

for i in test['fare']:

    if i == np.nan:

        tmp.append(i)

    else:

        tmp.append(np.round(i))

test['fare'] = tmp
dataset=pd.concat([train,test],join='inner')

dataset.head()
# fare column missing value processing

tmp = dataset[['pclass','fare']]

X=tmp[tmp['fare'].notnull()]

y=tmp[tmp['fare'].isnull()]

train_x = X.loc[:,X.columns.difference(['fare'])]

train_y = X['fare']

test_x = y.loc[:,y.columns.difference(['fare'])]

test_y = y['fare']

rf.fit(train_x,train_y)

pred = rf.predict(test_x)

test['fare'].fillna(pred[0],inplace=True)
test.drop(['sibsp','parch','ticket','name','passengerid'],axis=1,inplace=True)
test.head()
# 1. LabelEncoding

train2 = train.copy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

encoding = le.fit_transform(train2['title'])

train2['title'] = encoding

encoding2 = le.fit_transform(train2['embarked'])

train2['embarked'] = encoding2

encoding3 = le.fit_transform(train2['sex'])

train2['sex'] = encoding3

category_features = ['sex','title','embarked','age','family_size']

for i in category_features:

    train2[i] = train2[i].astype(int)
test2=test.copy()

encoding=le.fit_transform(test2['title'])

test2['title']=encoding

encoding2=le.fit_transform(test2['embarked'])

test2['embarked'] = encoding2

encoding3 = le.fit_transform(test['sex'])

test2['sex'] = encoding3

category_features = ['sex','embarked','title','age','family_size']

for i in category_features:

    test2[i] = test2[i].astype(int)
train_x=train2.drop('survived',axis=1)

train_y=train2['survived']

test_x=test2
# model

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.svm import SVC

# parameter tunning

from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier()

ada = AdaBoostClassifier()

et = ExtraTreesClassifier()

gra = GradientBoostingClassifier()

xgb = XGBClassifier()

svm = SVC()
rfc_parm={"max_depth":[None,5,10],'n_estimators':[200],'max_features':[None,'sqrt','log2']}

ada_parm = {'n_estimators':[200],'random_state':[42,56],'learning_rate':[4.75,5.75]}

et_parm = {'n_jobs': [-1],'n_estimators':[200],"max_depth":[None,5,10],

          'criterion':['gini','entropy'],'max_features':[None,'sqrt','log2']}

gra_parm = {'n_estimators':[200],'max_depth':[None,5,10]}

xgb_parm = {'max_depth':[3,5,10],'gamma':[0,0.5,1],'max_features':['auto','log','log2'],'eta':[0.5,0.1,0.3]}

svm_parm = {'gamma':['scale','auto'],'degree':[5,10],'kernel':['linear','rbf'],'C':[0.025,0.5]}

rfc_cv=GridSearchCV(rf,rfc_parm) 

ada_cv=GridSearchCV(ada,ada_parm) 

et_cv=GridSearchCV(et,et_parm) 

gra_cv=GridSearchCV(gra,gra_parm) 

xgb_cv=GridSearchCV(xgb,xgb_parm) 

svm_cv=GridSearchCV(svm,svm_parm) 
test_y= submission['Survived'].astype(int)
model = [rfc_cv,ada_cv,et_cv,gra_cv,xgb_cv,svm_cv,rfc,ada,et,gra,xgb,svm]

name = ['rfc_cv','ada_cv','et_cv','gra_cv','xgb_cv','svm_cv','rfc','ada','et','gra','xgb','svm']

score_table = pd.DataFrame(columns=['Model','score'])

index=0

for i,k in zip(model,name):

    i.fit(train_x,train_y)

    score_table.loc[index,'Model'] = k

    score_table.loc[index,'score'] = i.score(test_x,test_y)

    index+=1

score_table.sort_values(by='score',ascending=True)
sns.barplot(data=score_table.sort_values(by='score'),x='Model',y='score')

plt.xticks(rotation=30,ha='right')

plt.title('Model accuracy score')
train3 = train.copy()

test3 = test.copy()
ohe_sex = ohe_trans(train3,'sex')

ohe_cabin = ohe_trans(train3,'cabin')

ohe_embarked = ohe_trans(train3,'embarked')

ohe_title = ohe_trans(train3,'title')

ohe_isalone = ohe_trans(train3,'isalone')

train3 = pd.concat([train3,ohe_sex,ohe_cabin,ohe_embarked,ohe_title,ohe_isalone],axis=1)
train3.drop(['sex','cabin','embarked','title','isalone'],axis=1,inplace=True)
ohe_sex = ohe_trans(test3,'sex')

ohe_cabin = ohe_trans(test3,'cabin')

ohe_embarked = ohe_trans(test3,'embarked')

ohe_title = ohe_trans(test3,'title')

ohe_isalone = ohe_trans(test3,'isalone')

test3 = pd.concat([test3,ohe_sex,ohe_cabin,ohe_embarked,ohe_title,ohe_isalone],axis=1)
test3.drop(['sex','cabin','embarked','title','isalone'],axis=1,inplace=True)
print(train3.columns)

print(test3.columns)

# Since the number of columns of two data is different, add one column.

for i in range(test3.shape[0]):

    test3.loc[i,'i1'] = 0
train_x_ohe=train3.drop('survived',axis=1)

train_y_ohe=train3['survived']

test_x_ohe=test3

model = [rfc_cv,ada_cv,et_cv,gra_cv,xgb_cv,svm_cv,rfc,ada,et,gra,xgb,svm]

name = ['rfc_cv','ada_cv','et_cv','gra_cv','xgb_cv','svm_cv','rfc','ada','et','gra','xgb','svm']

score_table2 = pd.DataFrame(columns=['Model','score'])

index=0

for i,k in zip(model,name):

    i.fit(train_x,train_y)

    score_table2.loc[index,'Model'] = k

    score_table2.loc[index,'score'] = i.score(test_x,test_y)

    index+=1

score_table2.sort_values(by='score',ascending=True)
sns.barplot(data=score_table2.sort_values(by='score'),x='Model',y='score')

plt.xticks(rotation=30,ha='right')

plt.title('Model accuracy score')
fig,axes = plt.subplots(ncols=2,figsize=(12,8))

sns.barplot(data=score_table.sort_values(by='score'),x='Model',y='score',ax=axes[0])

axes[0].set(title = 'LabelEncoding Model score')

axes[0].tick_params(labelrotation=30)

sns.barplot(data=score_table2.sort_values(by='score'),x='Model',y='score',ax=axes[1])

axes[1].set(title = 'OneHotEncoding Model score')

axes[1].tick_params(labelrotation=30)
print('''LabelEncoding mean_score : {}

OneHotEncoding mean_score : {}'''.format(np.mean(score_table['score']),np.mean(score_table2['score'])))
from sklearn.model_selection import StratifiedKFold,cross_val_score
kfold = StratifiedKFold(n_splits=10)
general_model = [rfc,ada,et,gra,xgb,svm]

general_name = ['rfc','ada','et','gra','xgb','svm']

ge_result = []

ge_means = []

df=pd.DataFrame(columns = ['model','score'])



for i in general_model:

    ge_result.append(cross_val_score(i,train_x,y=train_y,scoring='accuracy',cv=kfold,n_jobs=4))

for j in ge_result: 

    ge_means.append(np.mean(j))

idx = 0

for i in range(len(ge_means)): # before parameter tunning

    df.loc[idx,'score'] = ge_means[i]

    df.loc[idx,'model'] = general_name[i]

    idx += 1



cv_model = [rfc_cv,ada_cv,et_cv,gra_cv,xgb_cv,svm_cv]

cv_name = ['rfc_cv','ada_cv','et_cv','gra_cv','xgb_cv','svm_cv']



for i in range(len(cv_model)): # after parameter tunning

    cv_model[i].fit(train_x,train_y)

    df.loc[idx,'model'] = cv_name[i]

    df.loc[idx,'score'] = cv_model[i].best_score_

    idx += 1

sns.barplot(data=df.sort_values(by='score'), x='model',y='score')

plt.xticks(rotation=30,ha='right')

plt.title('Cross validation scores')
et_cv_best = et_cv.best_estimator_

gra_cv_best = gra_cv.best_estimator_

xgb_cv_best = xgb_cv.best_estimator_

# extract more than 0.8

df = df.sort_values(by='score')

df[df['score']>=0.80].model
# feature_importances

# best_estimator_.feature_importances_ <- grid_search

final_model = [rfc,et,ada,xgb,gra,gra_cv_best,et_cv_best,xgb_cv_best]

final_model_name = ['rfc','et','ada','xgb','gra','gra_cv','et_cv','xgb_cv']

fig,(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(8,1,figsize=(15,20))

for i,j,k in zip(final_model,final_model_name,[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]):

    feat_importance=pd.Series(i.feature_importances_,index=train_x.columns)

    feat_importance.nlargest(5).plot(kind='barh',color=['r','c','m','y','g'], ax=k)

    k.set(title='{} feature importances'.format(j))

    plt.tight_layout()
# ('xgb',xgb.get_params),('gra',gra.get_params)

# ('rfc',rfc.estimators_),('et',et.estimators_),('ada',ada.estimators_)
from sklearn.ensemble import VotingClassifier

votingC = VotingClassifier(estimators=[('et_cv', et_cv_best),('xgb_cv',xgb_cv_best),('gbc_cv',gra_cv_best)], voting='soft', n_jobs=4)

votingC.fit(train_x,train_y)

test_le = pd.Series(votingC.predict(test_x), name="Survived")
passengerid = submission['PassengerId']

results = pd.concat([passengerid,test_le],axis=1)

results.to_csv('label_titanic.csv',index=False)