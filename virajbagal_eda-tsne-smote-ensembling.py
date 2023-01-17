

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import os

from sklearn.preprocessing import StandardScaler

from scipy.stats import kurtosis,skew

from sklearn.manifold import TSNE

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV

from sklearn.svm import SVC

from sklearn.metrics import average_precision_score,make_scorer

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier,AdaBoostClassifier,RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from scipy.stats import zscore

from xgboost import XGBClassifier

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))



data=pd.read_csv('../input/creditcard.csv')

data.head()
data.shape
data.info()
data.isnull().sum()
data.describe()
f,ax=plt.subplots(7,4,figsize=(15,15))

for i in range(28):

    sns.distplot(data['V'+str(i+1)],ax=ax[i//4,i%4])

    



plt.tight_layout()

plt.show()
stats=pd.DataFrame()

cols=[col for col in data.columns[1:29]]

mean=data[cols].mean(axis=0)

std=data[cols].std(axis=0)

max_val=data[cols].max(axis=0)

min_val=data[cols].min(axis=0)

skew=data[cols].skew(axis=0)

kurt=data[cols].kurt(axis=0)

stats['mean']=mean

stats['std']=std

stats['max']=max_val

stats['min']=min_val

stats['skew']=skew

stats['kurt']=kurt

stats.index=cols

x_ticks=np.arange(1,29,1)

f,ax=plt.subplots(2,3,figsize=(15,8))

for i in range(6):

    ax[i//3,i%3].plot(x_ticks,stats.iloc[:,i].values,'b.')

    ax[i//3,i%3].set_title(stats.columns[i])

    

plt.tight_layout()

plt.show()
print(stats.loc[['V16','V18','V19'],:])
plt.figure(figsize=(15,10))

sns.heatmap(data.corr())

plt.show()
sns.countplot(data['Class'])

print((data['Class'].value_counts()/data.shape[0])*100)
X=data.drop(['Class','Time'],axis=1)

Y=data['Class']

train_X,test_X,train_y,test_y=train_test_split(X,Y,random_state=5,test_size=0.2)
sc=StandardScaler()

train_X=sc.fit_transform(train_X)

test_X=sc.transform(test_X)

train_X=pd.DataFrame(train_X,columns=X.columns)

test_X=pd.DataFrame(test_X,columns=X.columns)
sm=SMOTE(random_state=5)

train_X_res,train_y_res=sm.fit_sample(train_X,train_y)

train_X_res=pd.DataFrame(train_X_res,columns=train_X.columns)

train_y_res=pd.Series(train_y_res,name='Class')
train=pd.concat([train_X_res,train_y_res],axis=1)

fraud=train[train['Class']==1].sample(2500)

non_fraud=train[train['Class']==0].sample(2500)

tsne_data=pd.concat([fraud,non_fraud],axis=0)

tsne_data_1=tsne_data.drop(['Class'],axis=1)

tsne=TSNE(n_components=2,random_state=5,verbose=1)

tsne_trans=tsne.fit_transform(tsne_data_1)
tsne_data['first_tsne']=tsne_trans[:,0]

tsne_data['second_tsne']=tsne_trans[:,1]

plt.figure(figsize=(15,10))

sns.scatterplot(tsne_data['first_tsne'],tsne_data['second_tsne'],hue='Class',data=tsne_data)

models=[SVC(probability=True),LogisticRegression(),LinearDiscriminantAnalysis(),DecisionTreeClassifier(),

       ExtraTreesClassifier(n_estimators=100),AdaBoostClassifier(n_estimators=100),RandomForestClassifier(n_estimators=100)]



model_names=['SVC','LR','LDA','DTC','ETC','ABC','RFC']

train_score=[]

score_1=[]

test_score=[]
skf=StratifiedKFold(n_splits=5,random_state=5)

def get_model(train_X,train_y,test_X,test_y,model):

    for train_index,val_index in skf.split(train_X,train_y):

        train_X_skf,val_X_skf=train_X.iloc[train_index,:],train_X.iloc[val_index,:]

        train_y_skf,val_y_skf=train_y.iloc[train_index],train_y.iloc[val_index]

        clf=model

        clf.fit(train_X_skf,train_y_skf)

        pred=clf.predict_proba(val_X_skf)[:,1]

        score=average_precision_score(val_y_skf,pred)

        score_1.append(score)

        

    train_score.append(np.mean(score_1))

    clf.fit(train_X,train_y)

    pred_prob=clf.predict_proba(test_X)[:,1]

    score_test=average_precision_score(test_y,pred_prob)

    test_score.append(score_test)

           
train_X_sam=train_X_res.sample(10000)

train_X_index=train_X_sam.index

train_y_sam=train_y_res[train_X_index]

train_X_sam.reset_index(drop=True,inplace=True)

train_y_sam.reset_index(drop=True,inplace=True)

test_X_sam=test_X.sample(1000)

test_X_index=test_X_sam.index

test_y_sam=test_y[test_X_index]

test_X_sam.reset_index(drop=True,inplace=True)

test_y_sam.reset_index(drop=True,inplace=True)
for model in models:

    get_model(train_X_sam,train_y_sam,test_X,test_y,model)
result=pd.DataFrame({'models':model_names,'train_score':train_score,

                    'test_score':test_score},index=model_names)



plt.figure(figsize=(10,6))

plt.subplot(1,2,1)

result['train_score'].plot.bar()

plt.title('Train Score')

plt.subplot(1,2,2)

result['test_score'].plot.bar()

plt.title('Test Score')

plt.tight_layout()

plt.show()
clf=LogisticRegression()

clf.fit(train_X_sam,train_y_sam)

lr_pred=clf.predict_proba(test_X)[:,1]



clf_2=AdaBoostClassifier()

clf_2.fit(train_X_sam,train_y_sam)

abc_pred=clf_2.predict_proba(test_X)[:,1]



clf_3=ExtraTreesClassifier()

clf_3.fit(train_X_sam,train_y_sam)

etc_pred=clf_3.predict_proba(test_X)[:,1]



clf_4=RandomForestClassifier()

clf_4.fit(train_X_sam,train_y_sam)

rfc_pred=clf_4.predict_proba(test_X)[:,1]



xgb=XGBClassifier()

xgb.fit(train_X_sam,train_y_sam)

xgb_pred=xgb.predict_proba(test_X)[:,1]
blending_pred=0.20*(lr_pred+etc_pred+rfc_pred+abc_pred+xgb_pred)

blending_score=average_precision_score(test_y,blending_pred)

print(blending_score)
