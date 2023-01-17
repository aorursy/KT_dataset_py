# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score



from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc

from sklearn.linear_model import LogisticRegression

import xgboost as xgb

import time



from xgboost.sklearn import XGBClassifier

from sklearn import cross_validation, metrics   #Additional scklearn functions

from sklearn.grid_search import GridSearchCV ,RandomizedSearchCV  #Perforing grid search

from sklearn.model_selection import StratifiedKFold

import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4





train = pd.read_csv('../input/train.csv', na_values=-1)

test = pd.read_csv('../input/test.csv', na_values=-1)
train.dtypes
test.head()
#features = train.drop(['id','target'], axis=1).values

targets = train.target.values
ax = sns.countplot(x = targets ,palette="Set1")

sns.set(font_scale=1.5)

ax.set_xlabel(' ')

ax.set_ylabel(' ')

fig = plt.gcf()

fig.set_size_inches(10,5)

ax.set_ylim(top=700000)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x()+ 0.3, p.get_height()+10000))



plt.title('Distribution of 595212 Targets')

plt.xlabel('Initiation of Auto Insurance Claim Next Year')

plt.ylabel('Frequency [%]')

plt.show()
sns.set(style="white")





# Compute the correlation matrix

corr = train.corr()





# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
for i,k in train.isnull().sum().iteritems():

    print (i,k/train.shape[0]*100)
for i,k in test.isnull().sum().iteritems():

    print (i,k/test.shape[0]*100)
unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(unwanted, axis=1)  

test = test.drop(unwanted, axis=1) 
train.isnull().sum()
q= pd.DataFrame()

q['train']= train.isnull().sum()

q['test'] = test.isnull().sum()

fig,ax = plt.subplots(figsize=(16,5))

q.plot(kind='bar',ax=ax)
def basic_details(df):

    b = pd.DataFrame()

    #b['Missing value'] = df.isnull().sum()

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(train)
'''# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897

def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)

 

def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score

'''
'''# More parameters has to be tuned. Good luck :)

params = {

    'min_child_weight': 7.0,

    'objective': 'binary:logistic',

    'max_depth': 4,

    'max_delta_step': 2,

    'colsample_bytree': 0.8,

    'subsample': 0.8,

    'eta': 0.02,

    'gamma': 0.0,

    'num_boost_round' : 700

    }'''
def missing_value(df):

    col = df.columns

    for i in col:

        if df[i].isnull().sum()>0:

            df[i].fillna(df[i].mode()[0],inplace=True)
missing_value(train)

missing_value(test)
cat_col = [col for col in train.columns if '_cat' in col]

print(cat_col)
for c in cat_col:

    train[c] = train[c].astype('uint8')

    test[c] = test[c].astype('uint8') 
bin_col = [col for col in train.columns if 'bin' in col]

print(bin_col)
for c in bin_col:

    train[c] = train[c].astype('uint8')

    test[c] = test[c].astype('uint8') 
def category_col(df):

    c_col = df.columns

    to_cat_col =[]

    for i in c_col:

        if df[i].nunique()<=104:

            to_cat_col.append(i)

    return to_cat_col





tot_cat_col = category_col(train)

other_cat_col = [c for c in tot_cat_col if c not in cat_col+ bin_col]

other_cat_col
ot_col = ['ps_ind_01','ps_ind_03','ps_ind_14','ps_ind_15','ps_car_11']

for c in ot_col:

    train[c] = train[c].astype('uint8')

    test[c] = test[c].astype('uint8') 
num_col = [c for c in train.columns if c not in tot_cat_col]

num_col.remove('id')

num_col
train['ps_reg_03'].describe()
#time consuming

#fig,ax = plt.subplots(2,2,figsize=(14,8))

#ax1,ax2,ax3,ax4 = ax.flatten()

#sns.distplot(train['ps_reg_03'],bins=100,color='red',ax=ax1)

#sns.boxplot(x ='ps_reg_03',y='target',data=train,ax=ax2)

#sns.violinplot(x ='ps_reg_03',y='target',data=train,ax=ax3)

#sns.pointplot(x= 'ps_reg_03',y='target',data=train,ax=ax4)
train['ps_car_12'].describe()
fig,ax = plt.subplots(2,2,figsize=(14,8))

ax1,ax2,ax3,ax4 = ax.flatten()

sns.distplot(train['ps_car_12'],bins=50,ax=ax1)

sns.boxplot(x='ps_car_12',y='target',data=train,ax=ax2)

sns.violinplot(x='ps_car_12',y='target',data=train,ax=ax3)

sns.pointplot(x='ps_car_12',y='target',data=train,ax=ax4)
train['ps_car_13'].describe()
train['ps_car_14'].describe()
def outlier(df,columns):

    for i in columns:

        quartile_1,quartile_3 = np.percentile(df[i],[25,75])

        quartile_f,quartile_l = np.percentile(df[i],[1,99])

        IQR = quartile_3-quartile_1

        lower_bound = quartile_1 - (1.5*IQR)

        upper_bound = quartile_3 + (1.5*IQR)

        print(i,lower_bound,upper_bound,quartile_f,quartile_l)

        df[i].loc[df[i] < lower_bound] = quartile_f

        df[i].loc[df[i] > upper_bound] = quartile_l

        

outlier(train,num_col)

outlier(test,num_col) 
X = train.drop(['target','id'],axis=1)

y = train['target'].astype('category')

x_test = test.drop('id',axis=1)
'''xg_cl = xgb.XGBClassifier(max_depth=4,learning_rate=0.1,n_estimators=500,objective='binary:logistic',

                          min_child_weight=1,scale_pos_weight=1)

param = {#'max_depth':[4,7,8],

         #'learning_rate':[0.01,0.03,0.1,0.3],

         #'min_child_weight':[5,6,7],

         #'reg_lambda':[0.1,0.5,1,1.3,1.7,2.5],

         #'reg_alpha':[1,1.5,3,8,10,12,15],

         #'gamma':[0,0.1,0.5,1,5,10],

         #'scale_pos_weight':[0.5,1,2,3,5,10],

        #'subsample':[0.7,0.8,0.9],

        #'colsample_bytree':[0.7,0.8,0.9]

        }

clf = GridSearchCV(xg_cl,param,scoring='roc_auc',refit=True)



clf.fit(X,y)

print('Best roc_auc: {:.4}, with best params: {}'.format(clf.best_score_, clf.best_params_)) '''
def runXGB(xtrain,xvalid,ytrain,yvalid,xtest,eta=0.1,num_rounds=100,max_depth=4):

    params = {

        'objective':'binary:logistic',        

        'max_depth':max_depth,

        'learning_rate':eta,

        'eval_metric':'auc',

        'min_child_weight':6,

        'subsample':0.8,

        'colsample_bytree':0.8,

        'seed':45,

        'reg_lambda':1.3,

        'reg_alpha':8,

        'gamma':10,

        'scale_pos_weight':1.6

        #'n_thread':-1

    }

    

    dtrain = xgb.DMatrix(xtrain,label=ytrain)

    dvalid = xgb.DMatrix(xvalid,label=yvalid)

    dtest = xgb.DMatrix(xtest)

    watchlist = [(dtrain,'train'),(dvalid,'test')]

    

    model = xgb.train(params,dtrain,num_rounds,watchlist,early_stopping_rounds=50,verbose_eval=50)

    pred = model.predict(dvalid,ntree_limit=model.best_ntree_limit)

    pred_test = model.predict(dtest,ntree_limit=model.best_ntree_limit)

    return pred_test,model
cv=2
kf = StratifiedKFold(n_splits=cv,random_state=45)

pred_test_full =0

cv_score = []

i=1

for train_index,test_index in kf.split(X,y):

    print('{} of KFold {}'.format(i,kf.n_splits))

    xtr,xvl = X.loc[train_index],X.loc[test_index]

    ytr,yvl = y[train_index],y[test_index]

    

    pred_test,xg_model = runXGB(xtr,xvl,ytr,yvl,x_test,num_rounds=100,eta=0.1)    

    pred_test_full += pred_test

    cv_score.append(xg_model.best_score)

    i+=1
print(cv_score)

print('Mean cv_score',np.mean(cv_score))
pred_xgb = pred_test_full/cv
fig,ax = plt.subplots(figsize=(14,10))

xgb.plot_importance(xg_model,ax=ax,height=0.8,color='r')

plt.show()
y_pred = pred_xgb

submit = pd.DataFrame({'id':test['id'],'target':y_pred})

submit.to_csv('result.csv',index=False)
'''logreg = LogisticRegression(class_weight='balanced')

param = {'C':[0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5,1]}

clf = GridSearchCV(logreg,param,scoring='roc_auc',refit=True,cv=3)

clf.fit(X,y)

print('Best roc_auc: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C'])) '''
kf = StratifiedKFold(n_splits=cv,random_state=45,shuffle=True)

pred_test_full=0

cv_score=[]

i=1

for train_index,test_index in kf.split(X,y):    

    print('\n{} of kfold {}'.format(i,kf.n_splits))

    xtr,xvl = X.loc[train_index],X.loc[test_index]

    ytr,yvl = y[train_index],y[test_index]

    

    lr = LogisticRegression(class_weight='balanced',C=0.1)

    lr.fit(xtr, ytr)

    pred_test = lr.predict_proba(xvl)[:,1]

    score = roc_auc_score(yvl,pred_test)

    print('roc_auc_score',score)

    cv_score.append(score)

    pred_test_full += lr.predict_proba(x_test)[:,1]

    i+=1
print('Confusion matrix\n',confusion_matrix(yvl,lr.predict(xvl)))

print('Cv',cv_score,'\nMean cv Score',np.mean(cv_score))
proba = lr.predict_proba(xvl)[:,1]

fpr,tpr, threshold = roc_curve(yvl,proba)

auc_val = auc(fpr,tpr)



plt.figure(figsize=(14,8))

plt.title('Reciever Operating Charactaristics')

plt.plot(fpr,tpr,'b',label = 'AUC = %0.2f' % auc_val)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')
y_pred = pred_test_full/cv