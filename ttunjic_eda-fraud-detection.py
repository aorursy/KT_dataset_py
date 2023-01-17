!pip install chart_studio
!pip install plotly==3.10.0
import numpy as np 

import pandas as pd 





import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

import xgboost as xgb

import lightgbm as lgb

import catboost

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score

import matplotlib.gridspec as gridspec

%matplotlib inline



# Standard plotly imports

#import plotly.plotly as py

#import plotly.graph_objs as go

#import plotly.tools as tls

#from plotly.offline import iplot, init_notebook_mode

#import cufflinks

#import cufflinks as cf

#import plotly.figure_factory as ff



# Using plotly + cufflinks in offline mode

#init_notebook_mode(connected=True)

#cufflinks.go_offline(connected=True)





import warnings

warnings.filterwarnings("ignore")



import gc

gc.enable()



import logging

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
folder_path = '../input/ieee-fraud-detection/'

ind = 'TransactionID'

train_identity = pd.read_csv(f'{folder_path}train_identity.csv')

train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv')

test_identity = pd.read_csv(f'{folder_path}test_identity.csv')

test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv')

sub = pd.read_csv(f'{folder_path}sample_submission.csv')

# let's combine the data and work with the whole dataset

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
train_transaction.head()
train_identity.head()
test_transaction.head()
train_identity.shape, train_transaction.shape
print('train_transaction has shape',train_transaction.shape)

print('train_identity has shape',train_identity.shape)

print('test_transaction has shape',test_transaction.shape)

print('test_identity has shape',test_identity.shape)
train_transaction['TransactionDT']
fig = plt.figure(figsize = (12,8))

ax = fig.gca()



plt.hist(train['TransactionDT'], label='train',bins=100);

plt.hist(test['TransactionDT'], label='test',bins=100);

plt.legend();

plt.title('Distribution of Transaction Dates');



   
del train_identity,train_transaction,test_identity,test_transaction

gc.collect()
def red_mem_usage(df, verbose=True):

    num = ['int16','int32','int64','float16','float32','float64']

    start_mem = df.memory_usage(deep = True).sum()/1024**2

    for col in df.columns:

        col_types = df[col].dtypes

        if col_types in num:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_types)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                else:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem=df.memory_usage(deep = True).sum()/1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

    

    
red_mem_usage(train)

red_mem_usage(test)
def analize(df):

    analysis = pd.DataFrame(df.dtypes,columns=['d_types'])

    analysis = analysis.reset_index()

    analysis = analysis.rename(columns={"index": "Col_name"})

    analysis['Missing_values'] = df.isnull().sum().values

    analysis['Unique_values'] = df.nunique().values

    return analysis
def overview(df):

    for col, values in df.iteritems():

        num_uniques = values.nunique()

        print ('{name}: {num_unique}'.format(name=col, num_unique=num_uniques))

        print (values.unique())

        print ('\n')

    return
train[['TransactionAmt']].describe()
train[['TransactionAmt']].astype('float32').describe()
plt.figure(figsize=(16,12))

plt.suptitle('Transaction Amount Distributions', fontsize=18)



plt.subplot(221)

d = sns.distplot(train['TransactionAmt'])

d.set_title("Transaction Amount Distribuition", fontsize=18)

d.set_xlabel("")

d.set_ylabel("Probability", fontsize=15)





plt.subplot(222)

d1 = sns.distplot(train[train['TransactionAmt'] <= 1000]['TransactionAmt'])

d1.set_title("Transaction Amount Distribuition <= 1000", fontsize=18)

d1.set_xlabel("")

d1.set_ylabel("Probability", fontsize=15)



plt.subplot(223)

l = sns.distplot(np.log(train['TransactionAmt']),color='r')

l.set_title("Transaction Amount (Log) Distribuition", fontsize=18)

l.set_xlabel("")

l.set_ylabel("Probability", fontsize=15)



plt.subplot(224)

l1 = sns.distplot(np.log(train[train['TransactionAmt']<=1000]['TransactionAmt']),color='r')

l1.set_title("Transaction Amount (Log) Distribuition <= 1000", fontsize=18)

l1.set_xlabel("")

l1.set_ylabel("Probability", fontsize=15)



plt.figure(figsize=(16,12))



plt.figure(figsize=(16,12))

plt.suptitle('Transaction Amount Distributions, isFraud==1', fontsize=18)



plt.subplot(221)

d = sns.distplot(train[train['isFraud']==1]['TransactionAmt'])

d.set_title("Transaction Amount Distribuition", fontsize=18)

d.set_xlabel("")

d.set_ylabel("Probability", fontsize=15)



plt.subplot(222)

d = sns.distplot(np.log(train[train['isFraud']==1]['TransactionAmt']),color='r')

d.set_title("Transaction Amount (Log) Distribuition", fontsize=18)

d.set_xlabel("")

d.set_ylabel("Probability", fontsize=15)



plt.figure(figsize=(16,12))

plt.suptitle('Transaction Amount Distributions, isFraud==0', fontsize=18)



plt.subplot(221)

d = sns.distplot(train[train['isFraud']==0]['TransactionAmt'],color='b')

d.set_title("Transaction Amount Distribuition", fontsize=18)

d.set_xlabel("")

d.set_ylabel("Probability", fontsize=15)



plt.subplot(222)

d = sns.distplot(np.log(train[train['isFraud']==0]['TransactionAmt']),color='r')

d.set_title("Transaction Amount (Log) Distribuition", fontsize=18)

d.set_xlabel("")

d.set_ylabel("Probability", fontsize=15)
train[train['isFraud']==0]['TransactionAmt'].values.mean()
train[train['isFraud']==1]['TransactionAmt'].values.mean()
train[train['isFraud']==0]['isFraud'].count()/train['isFraud'].count()
analize(train[['dist1','dist2']])
print("{0:.2f}".format((352271/590541)*100),'% of missing values in dist1 column and',"{0:.2f}".format((552913/590541)*100),'% of missing values in dist 2 column')
plt.figure(figsize=(16,12))

plt.suptitle('dist1 and dist2',fontsize=18)



a=train[['dist1']].dropna(axis=0)

b=train[['dist2']].dropna(axis=0)



plt.subplot(221)

d = sns.distplot(a,color='b')

d.set_title("dist1 Distribuition", fontsize=18)

d.set_xlabel("")

d.set_ylabel("Probability", fontsize=15)



plt.subplot(222)

d2 = sns.distplot(b,color='r')

d2.set_title("dist2 Distribution", fontsize=18)

d2.set_xlabel("")

d2.set_ylabel("Probability", fontsize=15)

analize(train[['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']])
overview(train[['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']])
train[['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']].astype('float32').describe()
analize(train[['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15']])
train[['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15']][:20]
train[['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15']].astype('float32').describe()
pd.set_option('display.max_columns',400)

v_col = [c for c in train if c[0] == 'V']

train[v_col].head()
id_col = ['id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11']

train[id_col].astype('float32').describe()
analize(train[id_col])
plt.figure(figsize=(35, 12))

features = list(train[id_col])

uniques = [len(train[col].unique()) for col in features]

sns.set(font_scale=1.2)

ax = sns.barplot(features, uniques, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TRAIN')

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            uniq,

            ha="center") 
test.columns
analize(test)
train.head()
for i in range(39):

    if i<9:

        test=test.rename(columns={"id-0"+str(i+1): "id_0"+str(i+1)})

    test=test.rename(columns={"id-"+str(i+1): "id_"+str(i+1)})

test.head()
plt.figure(figsize=(35, 12))

features = list(test[id_col])

uniques = [len(test[col].unique()) for col in features]

sns.set(font_scale=1.2)

ax = sns.barplot(features, uniques, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TEST')

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            uniq,

            ha="center") 
l=[]

for i in range(12,39):

    l.append('id_'+str(i))

train[l].head()
n=[i for i in l if train[i].dtype=='float16']

train[n]=train[n].astype('float32')

train[n].describe()

c = [k for k in l if train[k].dtype=='object']

train[c].describe()
analize(train[l])
overview(train[l])
for i in n:

    try:

        train.set_index('TransactionDT')[i].plot(style='.', title=i, figsize=(15, 3))

        test.set_index('TransactionDT')[i].plot(style='.', title=i, figsize=(15, 3))

        plt.show()

    except TypeError:

        pass
cols = ['TransactionDT'] + n

plt.figure(figsize=(15,15))

sns.heatmap(train[cols].corr(), cmap='RdBu_r', annot=True, center=0.0)

plt.title('ID')

plt.show()
enc_c = ['id_12','id_15','id_16','id_27','id_28','id_29','id_34','id_35','id_36','id_37','id_38']

nenc_c=[k for k in c if k not in enc_c]

dc = {'Unknown':-1,'NotFound':0,'Found':1,'New':2,'F':0,'T':1,'match_status:2':2, 'match_status:1':1, 'match_status:0':0, 'match_status:-1':-1}

for i in enc_c:

    train[i]=train[i].map(dc)
cols = ['TransactionDT'] + enc_c

plt.figure(figsize=(15,15))

sns.heatmap(train[cols].corr(), cmap='RdBu_r', annot=True, center=0.0)

plt.title('ID')

plt.show()
for i in nenc_c:

    plt.figure(figsize=(80,30))



    train[i]=train[i].fillna('Missing')

    features = list(train[i].unique()[:20])

    #if you want to see 10 most frequent values 

    #features = train['DeviceInfo'].value_counts()[:10].index.tolist()

    uniques = [(train[i]==col).sum() for col in features]

    sns.set(font_scale=2)

    ax = sns.barplot(features,uniques, log=True)

    ax.set(xlabel='Feature', ylabel='log(unique count)', title=i)

    for p, uniq in zip(ax.patches, uniques):

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 10,

                uniq,

                ha="center") 
overview(train[['ProductCD']])
plt.figure(figsize=(30,15))

i='ProductCD'

#train[i]=train[i].fillna('Missing')

features = list(train[i].unique())

uniques = [(train[i]==col).sum() for col in features]

sns.set(font_scale=1)

ax = sns.barplot(features,uniques, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title=i)

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

                height + 10,

                uniq,

                ha="center") 
overview(train[['DeviceType']])
plt.figure(figsize=(30,15))

i='DeviceType'

train[i]=train[i].fillna('Missing')

features = list(train[i].unique())

uniques = [(train[i]==col).sum() for col in features]

sns.set(font_scale=1)

ax = sns.barplot(features,uniques, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title=i)

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

                height + 10,

                uniq,

                ha="center") 
overview(train[['DeviceInfo']])
plt.figure(figsize=(30,15))

i='DeviceInfo'

train[i]=train[i].fillna('Missing')

features = train['DeviceInfo'].value_counts()[:10].index.tolist()

uniques = [(train[i]==col).sum() for col in features]

sns.set(font_scale=1)

ax = sns.barplot(features,uniques, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title=i)

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

                height + 10,

                uniq,

                ha="center") 
train['DeviceInfo'].value_counts()[:10].index.tolist()
new = ['card1','card2','card3','card4','card5','card6','M1','M2','M3','M4','M5','M6','M7','M8','M9']

for i in new:

    plt.figure(figsize=(25,10))



    train[i]=train[i].fillna('Missing')

    features = list(train[i].unique()[:20])

    #if you want to see 10 most frequent values 

    #features = train['DeviceInfo'].value_counts()[:10].index.tolist()

    uniques = [(train[i]==col).sum() for col in features]

    sns.set(font_scale=2)

    ax = sns.barplot(features,uniques, log=True)

    ax.set(xlabel='Feature', ylabel='log(unique count)', title=i)

    for p, uniq in zip(ax.patches, uniques):

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 10,

                uniq,

                ha="center") 
overview(train[['addr1','addr2']])
analize(train[['addr1','addr2']])
train[['addr1','addr2']].describe()
(train['addr2']==87).sum()
analize(train[['P_emaildomain','R_emaildomain']])
overview(train[['P_emaildomain','R_emaildomain']])
for i in ['P_emaildomain','R_emaildomain']:

    plt.figure(figsize=(25,10))



    train[i]=train[i].fillna('Missing')

    #features = list(train[i].unique()[:20])

    #if you want to see 10 most frequent values 

    features = train[i].value_counts()[:10].index.tolist()

    uniques = [(train[i]==col).sum() for col in features]

    sns.set(font_scale=2)

    ax = sns.barplot(features,uniques, log=True)

    ax.set(xlabel='Feature', ylabel='log(unique count)', title=i+' most frequent email adresses')

    for p, uniq in zip(ax.patches, uniques):

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 10,

                uniq,

                ha="center") 
for i in ['P_emaildomain','R_emaildomain']:

    plt.figure(figsize=(25,10))



    train[i]=train[i].fillna('Missing')

    #features = list(train[i].unique()[:20])

    #if you want to see 10 most frequent values 

    features = train[i].value_counts()[-10:].index.tolist()

    uniques = [(train[i]==col).sum() for col in features]

    sns.set(font_scale=2)

    ax = sns.barplot(features,uniques, log=True)

    ax.set(xlabel='Feature', ylabel='log(unique count)', title=i+' least frequent email adresses')

    for p, uniq in zip(ax.patches, uniques):

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 10,

                uniq,

                ha="center") 
for i in ['P_emaildomain','R_emaildomain']:

    plt.figure(figsize=(25,10))



    #train[i]=train[i].fillna('Missing')

    #features = list(train[i].unique()[:20])

    #if you want to see 10 most frequent values 

    features = (train[train.iloc[:]['addr2']== 87]['P_emaildomain']).value_counts(sort=True)[:10].index.tolist()

    uniques = [(train[i]==col).sum() for col in features]

    sns.set(font_scale=2)

    ax = sns.barplot(features,uniques, log=True)

    ax.set(xlabel='Feature', ylabel='log(unique count)', title=i+' with addr2==87')

    for p, uniq in zip(ax.patches, uniques):

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 10,

                uniq,

                ha="center") 