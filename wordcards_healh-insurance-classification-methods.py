# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings; warnings.simplefilter('ignore')
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import power_transform
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
train = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv').set_index('id')
test = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv').set_index('id')
print('train :\t',train.shape)
print('test :\t',test.shape)
train.head()
train.isnull().sum()
test.isnull().sum()
cat_df = pd.DataFrame(columns=['train_nunique','test_nunique','n_values_only_in_test'])
cat_col = ['Gender','Region_Code','Vehicle_Age','Policy_Sales_Channel']

for c in cat_col:
    cat_df.at[c,'train_nunique'] = train[c].nunique()
    cat_df.at[c,'test_nunique'] = test[c].nunique()
    cat_df.at[c, 'n_values_only_in_test'] = len(
        [v for v in test[c].unique() if v not in train[c].unique()])

cat_df
def plot_cat(col,height,orderlist=None):
    fig,ax = plt.subplots(1,2,sharey=True,figsize=(14,height))
    sns.countplot(y=train[col],orient='h',order=orderlist,ax=ax[0])
    sns.pointplot(y=col,x='Response',data=train,orient='h',ax=ax[1])
    ax[1].set_xlim((0,0.3));
plot_cat('Gender',2)
plot_cat('Driving_License',2)
plot_cat('Previously_Insured',2)
plot_cat('Vehicle_Age',3,orderlist=['< 1 Year','1-2 Year','> 2 Years'])
plot_cat('Vehicle_Damage',2)
fig,ax = plt.subplots(3,2,figsize=(14,12))
sns.distplot(train['Age'],kde=False,ax=ax[0,0])
sns.violinplot(x='Age',y='Response',data=train,orient='h',ax=ax[0,1])
sns.distplot(train['Vintage'],kde=False,ax=ax[1,0])
sns.violinplot(x='Vintage',y='Response',data=train,orient='h',ax=ax[1,1])
sns.distplot(train['Annual_Premium'],kde=False,ax=ax[2,0])
sns.violinplot(x='Annual_Premium',y='Response',data=train,orient='h',ax=ax[2,1]);
y = train['Response']

train1 = train.drop('Response', axis=1)
test1 =test.copy()
train1['status'], test1['status'] = 'train', 'test'
allx = pd.concat([train1, test1])

allx['sex'] = allx['Gender'].map({'Male':0, 'Female':1})
allx['vehicle_age'] = allx['Vehicle_Age'].map({'< 1 Year':0, '1-2 Year':1, '> 2 Years':2})
allx['vegicle_damage'] = allx['Vehicle_Damage'].map({'No':0, 'Yes':1})

# Age, Annual_Premium -> Box-Cox transform
allx['age'], allx['premium'] = 0, 0
allx[['age', 'premium']] = power_transform(allx[['Age', 'Annual_Premium']], method='box-cox')

# Region_Code, Policy_Sales_Channel -> Count Encoding
region_dic = allx['Region_Code'].value_counts().to_dict()
max_region = allx['Region_Code'].value_counts().max()
allx['region'] = allx['Region_Code'].map(region_dic) / max_region

channel_dic = allx['Policy_Sales_Channel'].value_counts().to_dict()
max_channel = allx['Policy_Sales_Channel'].value_counts().max()
allx['channel'] = allx['Policy_Sales_Channel'].map(channel_dic) / max_channel

allx.drop(['Gender','Age','Region_Code','Vehicle_Age','Vehicle_Damage','Annual_Premium',
           'Policy_Sales_Channel','Vintage'], axis=1, inplace=True)
x = allx[allx['status']=='train'].drop('status', axis=1)
test_x = allx[allx['status']=='test'].drop('status', axis=1)

x_train, x_valid, y_train, y_valid = train_test_split(x,y,test_size=0.3,random_state=42)
def auc_validation(clf):
    return roc_auc_score(y_valid, clf.fit(x_train, y_train).predict_proba(x_valid)[:,1])

LRC = LogisticRegression()
RFC = RandomForestClassifier(random_state=0)
GBC = GradientBoostingClassifier(random_state=0)
XGB = XGBClassifier(random_state=0)
LGB = LGBMClassifier(random_state=0)
print('LRC:\t', auc_validation(LRC))
print('RFC:\t', auc_validation(RFC))
print('GBC:\t', auc_validation(GBC))
print('XGB:\t', auc_validation(XGB))
print('LGB:\t', auc_validation(LGB))