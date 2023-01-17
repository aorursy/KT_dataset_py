# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
%matplotlib inline
train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
train_df.set_index('PassengerId',inplace=True)
test_df.set_index('PassengerId',inplace=True)
dataset=[train_df,test_df]
train_df.head()
print(train_df.shape,test_df.shape)
train_df.isnull().sum().sort_values(ascending=False)
for data in dataset:
    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)
    data['Age'].fillna(data['Age'].median(),inplace=True)
    data['Fare'].fillna(data['Fare'].mode()[0],inplace=True)
    data.drop(['Cabin','Ticket'],axis=1,inplace=True)
print(train_df.shape,test_df.shape)
train_df.info()
#creat feature
for data in dataset:
    data['Agebin']=pd.cut(data['Age'].astype(int),4)
    data['Farebin']=pd.cut(data['Fare'],[0.,20.,50.,100.,600.],right=False)
    data['FamilySize']=data['SibSp']+data['Parch']+1
    data['IsAlone']=1
    data['IsAlone'].loc[data['FamilySize']>1]=0
    data['Title']=data['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]
#train_df['Title'].value_counts()
stat_min=10
title_num=(train_df['Title'].value_counts()<10)
train_df['Title']=train_df['Title'].apply(lambda x: 'Misc' if title_num[x]==True else x)
title_num=train_df['Title'].value_counts()
test_df['Title']=test_df['Title'].apply(lambda x: x if x in title_num else 'Misc')

test_df['Farebin'].value_counts()
test_df.isnull().sum().sort_values(ascending=False)
label=LabelEncoder()
for data in dataset:
    data['Agebin_Code']=label.fit_transform(data['Agebin'])
    data['Farebin_Code']=label.fit_transform(data['Farebin'])
    data['Title_Code']=label.fit_transform(data['Title'])
    data['Embarked_Code']=label.fit_transform(data['Embarked'])
    data['Sex_Code']=label.fit_transform(data['Sex'])
train_df.head(2)
y_train=train_df['Survived']
x_train=train_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','FamilySize','IsAlone']]
x_train_c=train_df[['Pclass','Sex_Code','Age','SibSp','Parch','Fare','Agebin_Code','Farebin_Code','FamilySize','IsAlone','Title_Code','Embarked_Code']]
x_test_c=test_df[['Pclass','Sex_Code','Age','SibSp','Parch','Fare','Agebin_Code','Farebin_Code','FamilySize','IsAlone','Title_Code','Embarked_Code']]
dummy=['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']
x_train_dummy=pd.get_dummies(train_df[dummy])
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x_train_c,y_train,random_state=0)
train_x_dummy,test_x_dummy,train_y_dummy,test_y_dummy=train_test_split(x_train_dummy,y_train,random_state=0)
print(x_train_c.shape,train_x.shape,test_x.shape)
print(y_train.shape,train_y.shape,test_y.shape)
print(train_x_dummy.shape,test_x_dummy.shape,train_y_dummy.shape,test_y_dummy.shape)
for x in dummy:
    if train_df[x].dtype!='float64':
        print('Survived Correlation by:%s' %x)
        print(train_df[['Survived',x]].groupby(['Survived',x])['Survived'].count())
import seaborn as sns
plt.figure(figsize=(16,9))
grid=sns.FacetGrid(train_df,'Survived')
grid.map(plt.hist,'Age')
grid.add_legend()
#plt.hist(x_train_c['Pclass'],y_train)
grid=sns.FacetGrid(train_df,row='Sex',col='Pclass',hue='Survived')
grid.map(plt.hist,'Age',alpha=0.75)
grid.add_legend()
plt.figure(figsize=(16,9))
plt.subplot(131)
plt.hist(x=[train_df[train_df['Survived']==1]['Fare'],train_df[train_df['Survived']==0]['Fare']],stacked=True,color=['red','blue'],label=['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(132)
plt.hist(x=[train_df[train_df['Survived']==1]['Age'],train_df[train_df['Survived']==0]['Age']],stacked=True,color=['red','blue'],label=['Survived','Dead'])
plt.title('Age Histogram by Surviva')
plt.xlabel('Age')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(133)
plt.hist(x=[train_df[train_df['Survived']==1]['FamilySize'],train_df[train_df['Survived']==0]['FamilySize']],stacked=True,color=['red','blue'],label=['Survived','Dead'])
plt.title('FamilySize Histogram by Surviva')
plt.xlabel('FamilySize')
plt.ylabel('# of Passengers')
plt.legend()
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
correlation_heatmap(train_df)
from sklearn import svm,tree,linear_model,neighbors,naive_bayes
from sklearn.ensemble import RandomForestClassifier
#svc=svm.SVC()
#knn=neighbors.KNeighborsClassifier()
#dtree=tree.DecisionTreeClassifier()
#lrc=linear_model.LogisticRegressionCV()
#rc=linear_model.RidgeClassifierCV()
#forest=RandomForestClassifier()
#bayes=naive_bayes.GaussianNB()
#models=[svc,knn,dtree,lrc,rc,forest,bayes]
x_train_c.info()
svc_grid={'C':[5,6,8],'kernel':['linear'],'gamma':[0.0000001,0.0000005,0.000001]}
knn_grid={'n_neighbors':[5,6,7,8,9],'algorithm':['kd_tree'],'weights':['distance']}
dtree_grid={'criterion':['gini'],'min_samples_split':[0.01,0.05,0.1],'random_state':[0]}
lrc_grid={'Cs': [10, 100, 1000]}
#rc_grid={'alphas':[(0.1, 1.0, 10.0)]}
forest_grid={'max_depth':[4,5,6,7,8],'n_estimators':[20,30,40,50,80,100],'criterion':['gini','entropy']}
bayes_grid={}
from sklearn.model_selection import cross_val_score,GridSearchCV,KFold
class grid():
    def __init__(self,model):
        self.model=model
    def grid_get(self,X,y,param_grid):
        grid_search=GridSearchCV(self.model,param_grid,cv=5,scoring='accuracy')
        grid_search.fit(X,y)
        print(grid_search.best_params_,grid_search.best_score_)
        return grid_search.best_params_

#grid_svc=grid(svm.SVC()).grid_get(x_train_c,y_train,svc_grid)
#grid_knn=grid(neighbors.KNeighborsClassifier()).grid_get(x_train_c,y_train,knn_grid)
#grid_forest=grid(RandomForestClassifier()).grid_get(x_train_c,y_train,forest_grid)
#grid_dtree=grid(tree.DecisionTreeClassifier()).grid_get(x_train_c,y_train,dtree_grid)
#grid_lrc=grid(linear_model.LogisticRegressionCV()).grid_get(x_train_c,y_train,lrc_grid)
#grid_rc=grid(linear_model.RidgeClassifierCV()).grid_get(x_train_c,y_train,rc_grid)
svc=svm.SVC(C=5, gamma=1e-05, kernel='linear')
knn=neighbors.KNeighborsClassifier(algorithm='kd_tree',n_neighbors=6,weights='distance')
dtree=tree.DecisionTreeClassifier(criterion='gini',min_samples_split=0.05,random_state= 0)
lrc=linear_model.LogisticRegressionCV(Cs=1000)
#rc=linear_model.RidgeClassifierCV(grid_rc)
forest=RandomForestClassifier(criterion='gini',max_depth=8,n_estimators=80)
bayes=naive_bayes.GaussianNB()
models=[svc,forest]
meta_model=knn
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,mod,meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)
        
    def fit(self,X,y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))
        
        for i,model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X,y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index,i] = renew_model.predict(X[val_index])
        
        self.meta_model.fit(oof_train,y)
        return self
    
    def predict(self,X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) 
                                      for single_model in self.saved_model]) 
        return self.meta_model.predict(whole_test)
    
    def get_oof(self,X,y,test_X):
        oof = np.zeros((X.shape[0],len(self.mod)))
        test_single = np.zeros((test_X.shape[0],5))
        test_mean = np.zeros((test_X.shape[0],len(self.mod)))
        for i,model in enumerate(self.mod):
            for j, (train_index,val_index) in enumerate(self.kf.split(X,y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index],y[train_index])
                oof[val_index,i] = clone_model.predict(X[val_index])
                test_single[:,j] = clone_model.predict(test_X)
            test_mean[:,i] = test_single.mean(axis=1)
        return oof, test_mean
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
#a=Imputer().fit_transform(x_train_c)
#b=Imputer().fit_transform(y_train)
scaler=MinMaxScaler()
x_fare=x_train_c['Fare'].reshape(-1,1)
x_fare=scaler.fit_transform(x_fare)
x_t_c=x_train_c.copy()
x_t_c['Fare']=x_fare
x_t_c.head()
a = Imputer().fit_transform(x_train_c)
b = Imputer().fit_transform(y_train.values.reshape(-1,1)).ravel()
#y.isnull().any()
#x.isnull().any()
stack_model=stacking(models,meta_model)
stack_model.fit(a,b)
result=stack_model.predict(x_test_c)
a.head()
result=np.c_[x_test_c.index,result]
result=pd.DataFrame(result,columns=['PassengerId','Survived'])
result=result.astype(int)
result.to_csv('Submission.csv',index=False)