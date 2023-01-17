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
train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')



print(train.columns)

print(test.columns)
PassengerId=test['PassengerId']
full_data=[train,test]

train['Name_length']=train['Name'].str.len()

test['Name_length']=test['Name'].str.len()

train['Has_Cabin']=train['Cabin'].apply(lambda x:0 if type(x)== float else 1)

test['Has_Cabin']=test['Cabin'].apply(lambda x:0 if type(x)== float else 1)

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp']+dataset['Parch']+1

for dataset in full_data:

    dataset['IsAlone']=0

    dataset.loc[dataset['FamilySize']==1,['IsAlone']]=1



for dataset in full_data:

    dataset['Embarked']=dataset['Embarked'].fillna('S')

for dataset in full_data:

    dataset['Fare']=dataset['Fare'].fillna(dataset['Fare'].median())

train['CategoricalFare']=pd.qcut(train['Fare'],4)

for dataset in full_data:

    age_avg=dataset['Age'].mean()

    age_std=dataset['Age'].std()

    age_null_count=dataset['Age'].isnull().sum()

    age_null_random_list=np.random.randint(age_avg-age_std,age_avg+age_std,age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])]=age_null_random_list

    dataset['Age']=dataset['Age'].astype(int)

train['CategoricalAge']=pd.cut(train['Age'],5)



import re

def get_title(name):

    title_search = re.search('([A-Za-z]+)\.',name)

    if title_search:

        return title_search.group(1)

    return ""

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')

    dataset['Title'] = dataset['Title'].replace('Ms','Miss')

    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

    
for dataset in full_data:

    dataset['Sex']=dataset['Sex'].map({'female':0,'male':1}).astype(int)

    title_mapping = {'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    dataset['Embarked']=dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

    dataset.loc[dataset['Fare']<=7.91,'Fare']=0

    dataset.loc[(dataset['Fare']>7.91)&(dataset['Fare']<=14.454),'Fare']=1

    dataset.loc[(dataset['Fare']>14.454)&(dataset['Fare']<=31.472),'Fare']=2

    dataset.loc[(dataset['Fare']>31.742)&(dataset['Fare']<=512.329),'Fare']=3

    dataset.loc[(dataset['Age'])<=16,'Age']=0

    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32),'Age']=1

    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48),'Age']=2

    dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64),'Age']=3

    dataset.loc[(dataset['Age']>64),'Age']=4
drop_elements = ['PassengerId','Name', 'Ticket','Cabin','SibSp']

train=train.drop(drop_elements,axis=1)

train = train.drop(['CategoricalAge','CategoricalFare'],axis=1)

test = test.drop(drop_elements,axis=1)
train.head(3)

import matplotlib.pyplot as plt

import seaborn as sns

colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features',y = 1.05,size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,linecolor='white',annot=True)
train.head(5)
g = sns.pairplot(train[[u'Survived',u'Pclass',u'Sex',u'Age',u'Parch',u'Fare',u'Embarked',u'FamilySize',u'Title']],

                hue='Survived',palette='seismic',size=1.2,diag_kind='kde',diag_kws=dict(shade=True),plot_kws=dict(s=10))

g.set(xticklabels=[])
from sklearn.model_selection import KFold

ntrain=train.shape[0]

ntest=test.shape[0]

SEED=0

NFOLDS = 5

kf=KFold(n_splits = NFOLDS,random_state=SEED)

class SklearnHelper(object):

    #首先调用python中类的构造函数__init__，这个构造函数有4个参数：self,clf,seed,params

    def __init__(self,clf,seed=0,params=None):

        params['random_state'] = seed

        self.clf = clf(**params)

    

    #设置类的训练函数（其实就是分类器的原生fit函数）

    def train(self,x_train,y_train):

        self.clf.fit(x_train,y_train)

    

    #返回预测结果（分类器的原生predict）

    def predict(self,x):

        return self.clf.predict(x)

    

    #这里我不是太懂，这个fit和train有什么不同

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    #返回训练好的模型的特征权重值

    def feature_importances(self,x,y):

        print(self.clf.fit(x,y).feature_importances_)
def get_oof(clf,x_train,y_train,x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS,ntest))

    

    for i, (train_index,test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        

        clf.train(x_tr,y_tr)

        

        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i,:] = clf.predict(x_test)

     

    oof_test[:] = oof_test_skf.mean(axis=0)

    

    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

#目测这个oof_train是交叉验证的出来的预测结果，oof_test是拿每一次训练后的模型，基于完整数据集去预测得到的完整结果
def yuan_oof(clf,x_train,y_train,x_test):

    oof_train=np.zeros((ntrain,))

    oof_test=np.zeros((ntest,))

    oof_predict_kf=np.zeros((NFOLDS,ntest))

    

    for i,(train_index,test_index) in enumerate(kf.split(x_train)):

        x_tr=x_train[train_index]#从训练集中划出来的训练集

        y_tr=y_train[train_index]

        x_te=x_train[test_index]#从训练集中划出来的验证集

        

        clf.train(x_tr,y_tr)

        

        oof_train[test_index]=clf.predict(x_te)

        oof_predict_kf[i,:] = clf.predict(x_test)

     

    oof_test=oof_predict_kf.mean(axis=0)

    

    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)
from sklearn.model_selection import KFold

import numpy as np

X = np.arange(12)

kf = KFold(n_splits=5,shuffle=False)

for train_index , test_index in kf.split(X):

    x_tr = X[train_index]

    print('train_index:%s , test_index: %s ' %(train_index,test_index))

    print('x_tr=%s' %(x_tr))
#random forest parameters

rf_params={

    'n_jobs':-1,

    'n_estimators':500,

    'warm_start':True,

    'max_depth':6,

    'min_samples_leaf':2,

    'max_features':'sqrt',

    'verbose':0

}



#extra trees parameters

et_params = {

    'n_jobs':-1,

    'n_estimators':500,

    'max_depth':8,

    'min_samples_leaf':2,

    'verbose':0

}



#adaboost parameters

ada_params = {

    'n_estimators':500,

    'learning_rate' :0.75

}



#gradient boosting parameters

gb_params = {

    'n_estimators':500,

    'max_depth':5,

    'min_samples_leaf':2,

    'verbose':0

}



#support vector classfier parameters

svc_params = {

    'kernel':'linear',

    'C':0.025

}
from sklearn.svm import SVC

from sklearn.ensemble import (GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier,ExtraTreesClassifier)



rf = SklearnHelper(clf=RandomForestClassifier,seed=SEED,params=rf_params)

et=SklearnHelper(clf=ExtraTreesClassifier,seed=SEED,params = et_params)

ada=SklearnHelper(clf=AdaBoostClassifier,seed=SEED,params = ada_params)

gb=SklearnHelper(clf=GradientBoostingClassifier,seed=SEED,params = gb_params)

svc=SklearnHelper(clf=SVC,seed=SEED,params = svc_params)



train.values
y_train=train['Survived'].ravel()

train=train.drop(['Survived'],axis=1)

x_train=train.values

x_test=test.values
#创建我们自己OOF函数的训练和测试预测。这些结果将会被当成新的特征

et_oof_train,et_oof_test = get_oof(et,x_train,y_train,x_test)#extra trees

rf_oof_train,rf_oof_test = get_oof(rf,x_train,y_train,x_test)#random forest

ada_oof_train,ada_oof_test = get_oof(ada,x_train,y_train,x_test)#adaboost

gb_oof_train,gb_oof_test = get_oof(gb,x_train,y_train,x_test)#gradient boost

svc_oof_train,svc_oof_test = get_oof(svc,x_train,y_train,x_test)#support vector classifier

print('Training is complete')
rf_feature=rf.feature_importances(x_train,y_train)

et_feature=et.feature_importances(x_train,y_train)

ada_feature=ada.feature_importances(x_train,y_train)

gb_feature=gb.feature_importances(x_train,y_train)

rf_features=[0.12394244, 0.1956606,  0.029608 ,  0.02207045, 0.07335707, 0.0238044,

 0.10945899, 0.06332547, 0.06709608, 0.01307129, 0.27860522]

et_features=[0.12649066, 0.38614875,0.02601249, 0.0161799,  0.03812365, 0.02930858,

 0.04713942, 0.08739047, 0.0442849,  0.02493595 ,0.17398523]

ada_features=[0.03,  0.014, 0.012, 0.066 ,0.038, 0.008, 0.694, 0.014, 0.05 , 0.004 ,0.07 ]

gb_features=[0.08333974, 0.01359581 ,0.0511851,  0.01170791 ,0.05705642, 0.02461628,

 0.17164743, 0.03973429, 0.1083766,  0.005577,   0.43316343]
cols=train.columns.values

feature_dataframe=pd.DataFrame({'features':cols,

'Random Forest feature importances':rf_features,

'Extra Trees  feature importances':et_features,

'AdaBoost feature importances':ada_features,

'Gradient Boost feature importances':gb_features})
import plotly.graph_objs as go

import plotly.offline as py
trace=go.Scatter(

    y=feature_dataframe['Random Forest feature importances'].values,

    x=feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['Random Forest feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data=[trace]



layout = go.Layout(

    autosize=True,

    title='Random Forest feature importances',

    hovermode='closest',

     yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig=go.Figure(data=data,layout=layout)

py.iplot(fig,filename='scatter2010')



trace=go.Scatter(

    y=feature_dataframe['Extra Trees  feature importances'].values,

    x=feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['Extra Trees  feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data=[trace]



layout = go.Layout(

    autosize=True,

    title='Extra Trees  feature importances',

    hovermode='closest',

     yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig=go.Figure(data=data,layout=layout)

py.iplot(fig,filename='scatter2010')



trace=go.Scatter(

    y=feature_dataframe['AdaBoost feature importances'].values,

    x=feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['AdaBoost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data=[trace]



layout = go.Layout(

    autosize=True,

    title='AdaBoost feature importances',

    hovermode='closest',

     yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig=go.Figure(data=data,layout=layout)

py.iplot(fig,filename='scatter2010')



trace=go.Scatter(

    y=feature_dataframe['Gradient Boost feature importances'].values,

    x=feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['Gradient Boost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data=[trace]



layout = go.Layout(

    autosize=True,

    title='Gradient Boost feature importances',

    hovermode='closest',

     yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig=go.Figure(data=data,layout=layout)

py.iplot(fig,filename='scatter2010')
feature_dataframe['mean']=feature_dataframe.mean(axis=1)
feature_dataframe
y=feature_dataframe['mean'].values

x=feature_dataframe['features'].values

data=[go.Bar(

    x=x,

    y=y,

    width = 0.5,

            marker=dict(

               color = feature_dataframe['mean'].values,

            colorscale='Portland',

            showscale=True,

            reversescale = False

            ),

            opacity=0.6

)]



layout=go.Layout(

    autosize=True,

    title='Barplots of mean feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)



fig=go.Figure(data=data,layout=layout)

py.iplot(fig,filename='bar-direct-labels')
base_predictions_train = pd.DataFrame(

{

    'RandomForest':rf_oof_train.ravel(),

    'ExtraTrees':et_oof_train.ravel(),

    'AdaBoost':ada_oof_train.ravel(),

    'GradientBoost':gb_oof_train.ravel()

}

)

base_predictions_train.head()
data=[

    go.Heatmap(

        z=base_predictions_train.astype(float).corr().values,

        x=base_predictions_train.columns.values,

        y=base_predictions_train.columns.values,

        colorscale='Viridis',

        showscale=True,

        reversescale=True

    )

]

py.iplot(data,filename='labelled-heatmap')
x_train=np.concatenate((et_oof_train,rf_oof_train,ada_oof_train,gb_oof_train,

                       svc_oof_train),axis=1)

x_test=np.concatenate((et_oof_test,rf_oof_test,ada_oof_test,gb_oof_test,

                       svc_oof_test),axis=1)
import xgboost as xgb

gbm=xgb.XGBClassifier(

    n_estimators = 2000,

    max_depth = 4,

    min_child_weight =2 ,

    gamma=0.9,

    subsample=0.8,

    colsample_bytree=0.8,

    objective='binary:logistic',

    nthread=-1,

    scale_pos_weight=1

).fit(x_train,y_train)

predictions=gbm.predict(x_test)
submission=pd.DataFrame({'PassengerId':PassengerId,'Survived':predictions})

submission.to_csv("my_submission.csv",index=False)