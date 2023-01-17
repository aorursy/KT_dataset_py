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


import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import StandardScaler

import eli5

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer,accuracy_score

from eli5.sklearn import PermutationImportance

from pdpbox import pdp,get_dataset,info_plots

from sklearn.feature_selection import RFECV

from scipy.stats import norm, skew

from scipy.optimize import curve_fit

import shap
data=pd.read_csv("../input/heart.csv")
data.sample(5)
print(data.shape)
data.info()
continuous=['age','trestbps','chol','thalach','oldpeak']



f,ax=plt.subplots(3,2,figsize=(10,10))

for i,feature in enumerate(continuous):

    (mu,sigma)=norm.fit(data[feature])

    alpha=skew(data[feature])

    sns.distplot(data[feature],fit=norm,ax=ax[i//2][i%2])

    ax[i//2][i%2].set_title('Distribution of {}'.format(feature))

    ax[i//2][i%2].legend(['$\mu=$ {:.2f}, $\sigma=$ {:.2f}, $\\alpha=$ {:.2f}'.format(mu,sigma,alpha)],loc='best')

    ax[i//2][i%2].set_ylabel('Frequency')



plt.tight_layout()

plt.show()
bins=[28,40,50,60,80]

names=['Young Adult','Adult','Adult 2','Old']

data['age']=pd.cut(data['age'],bins=bins,labels=names)

age_map={'Young Adult':'0','Adult':'1','Adult 2':'2','Old':'3'}

data['age']=data['age'].map(age_map)

data['age']=data['age'].astype('int64')

data['age'].value_counts().plot.bar()

plt.xlabel('Age Category')

plt.ylabel('Frequency')

plt.show()
f,ax=plt.subplots(2,2,figsize=(10,10))

continuous.remove('age')

for i,feature in enumerate(continuous):

    df=data.groupby('age')[feature].mean().reset_index()

    sns.lineplot(data=df,y=df[feature],x=df['age'],ax=ax[i//2][i%2])

    ax[i//2][i%2].set_title('{}'.format(feature))

    

plt.tight_layout()

plt.show()
for feature in continuous:

    sns.boxplot(x='target',y=feature,data=data)

    plt.show()
f,ax=plt.subplots(5,2,figsize=(12,12))



for i,feature in enumerate(['age','sex','cp','fbs','restecg','exang','slope','ca','thal']):

    sns.countplot(x=feature,data=data,hue='target',ax=ax[i//2,i%2])

    ax[i//2,i%2].set_title('Distribution of target wrt {}'.format(feature))

    ax[i//2,i%2].legend(loc='best')



plt.tight_layout()

plt.show()
X=data.drop('target',axis=1)

Y=data['target']



train_X,test_X,train_y,test_y=train_test_split(X,Y,random_state=1,test_size=0.2)



ss=StandardScaler()



train_X[continuous]=ss.fit_transform(train_X[continuous])

test_X[continuous]=ss.transform(test_X[continuous])

    
my_model=xgb.XGBClassifier(n_estimators=100).fit(train_X,train_y)



feat_imp=pd.DataFrame({'importances':my_model.feature_importances_})

feat_imp['features']=train_X.columns

feat_imp=feat_imp.sort_values(by='importances',ascending=False)





feat_imp=feat_imp.set_index('features')

feat_imp.plot.barh(title='Feature Importances',figsize=(10,10))

plt.xlabel('Feature Importance Score')

plt.show()
perm=PermutationImportance(my_model,random_state=1).fit(test_X,test_y)

eli5.show_weights(perm,feature_names=test_X.columns.tolist())
corr=data.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr,annot=True)

plt.show()

for features in test_X.columns:

    partial_graph=pdp.pdp_isolate(model=my_model,dataset=test_X,model_features=test_X.columns,feature=features)

    pdp.pdp_plot(partial_graph,features)

    plt.show()
explainer=shap.TreeExplainer(my_model)

shap_values=explainer.shap_values(test_X)

shap.summary_plot(shap_values,test_X)
pred=my_model.predict(test_X)



print('Accuracy:',accuracy_score(test_y,pred))
rfc=RandomForestClassifier(n_estimators=100)

acc=make_scorer(accuracy_score)

rfecv=RFECV(rfc,cv=5,scoring=acc,step=1)

rfecv=rfecv.fit(train_X,train_y)



print('Optimal number of features:',rfecv.n_features_)

print('\nBest features:',train_X.columns[rfecv.support_])
train_X1=rfecv.transform(train_X)

test_X1=rfecv.transform(test_X)



rfc_1=RandomForestClassifier(n_estimators=100,random_state=2).fit(train_X1,train_y)

pred_1=rfc_1.predict(test_X1)

print('Accuracy:',accuracy_score(test_y,pred_1))
plt.plot(range(1,len(rfecv.grid_scores_)+1),rfecv.grid_scores_)

plt.xlabel('Number of features')

plt.ylabel('Grid Scores')

plt.show()