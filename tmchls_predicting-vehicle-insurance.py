# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
train.head()
test=pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
test.head()
train.shape,test.shape
train.info()
test.info()
train.describe()
test.describe()
train.isnull().sum()
test.isnull().sum()
px.pie(train,'Gender')
px.pie(train,'Driving_License')
px.pie(train,'Previously_Insured')
#correcting vehicle_age values
train['Vehicle_Age'].replace({'< 1 Year':'less_than_one_year','> 2 Years':'more_than_two_years'},inplace=True)
test['Vehicle_Age'].replace({'< 1 Year':'less_than_one_year','> 2 Years':'more_than_two_years'},inplace=True)
px.pie(train,'Vehicle_Age')
px.pie(train,'Vehicle_Damage')
px.pie(train,'Response')
c_columns=['Gender','Vehicle_Age','Vehicle_Damage']
for i in c_columns:
    train[i]=train[i].astype('category')
    test[i]=test[i].astype('category')
train.info()
test.info()
plt.figure(figsize=(10,10))
sns.distplot(train['Age'])
plt.axvline(x=train['Age'].mean(),linestyle='--',c='r',label='Mean Age')
plt.xticks(range(0,100,10))
plt.legend()
sns.boxplot(train['Age'])
sns.boxplot(train['Vintage'])
sns.boxplot(train['Policy_Sales_Channel'])
plt.figure(figsize=(10,10))
sns.distplot(train['Annual_Premium'])
plt.axvline(x=train['Annual_Premium'].mean(),linestyle='--',c='r',label='Avg annual premium')
plt.xticks(range(0,600000,30000),rotation=90)
plt.legend()
sns.boxplot(train['Region_Code'])
plt.figure(figsize=(20,5))
sns.boxplot(train['Annual_Premium'])
sns.stripplot(data=train,y='Gender',x='Age',jitter=True)
sns.countplot('Response',hue='Gender',data=train)
plt.figure(figsize=(10,5))
sns.countplot('Driving_License',hue='Gender',data=train)
sns.stripplot(data=train,x='Annual_Premium',y='Vehicle_Age',jitter=True)
sns.countplot('Vehicle_Age',hue='Vehicle_Damage',data=train)
sns.countplot('Vehicle_Age',hue='Previously_Insured',data=train)
sns.countplot('Vehicle_Age',hue='Gender',data=train)
sns.lmplot(y='Vintage',x='Annual_Premium',col='Response',row='Gender',fit_reg=True,data=train)
from sklearn.preprocessing import OrdinalEncoder
o=OrdinalEncoder()
for i in c_columns:
    train[i]=o.fit_transform(train[[i]])
    test[i]=o.transform(test[[i]])
train.head()
test.head()
#printing correlation matrix
corr=train.iloc[:,:-1].corr()
top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(train[top_features].corr(),annot=True)
# function to remove those independent features which are correlated
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
correlation(train.iloc[:,:-1],0.5)
c=train.drop(columns={'id','Policy_Sales_Channel', 'Vehicle_Age','Vehicle_Damage'})
c.head()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV,train_test_split
X=c.drop(columns='Response',axis=1)
y=c['Response']
#splitting the data
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.4,stratify=y,random_state=0)
#scaling the data
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
#function defined to fit the model
def model(model):
    model.fit(x_train,y_train)
    y_t=model.predict(x_train)
    y_pred=model.predict(x_test)
    tr=str('Training score:') + str(roc_auc_score(y_train,y_t))
    te=str('Test score:') + str(roc_auc_score(y_test,y_pred))
    return tr,te
model(RandomForestClassifier(random_state=0))
model(DecisionTreeClassifier(random_state=0))
model(XGBClassifier(random_state=0))
model(LGBMClassifier(random_state=0))
model(KNeighborsClassifier())
r_params={
    'n_neighbors': np.arange(1,15),
    'leaf_size':np.arange(20,100,10)
}
from tpot import TPOTClassifier


tpot_classifier = TPOTClassifier(generations= 3, population_size= 12, offspring_size= 6,
                                 verbosity= 2, early_stop= 12,
                                 config_dict={'sklearn.neighbors.KNeighborsClassifier': r_params}, 
                                 cv = 4, scoring = 'roc_auc')
tpot_classifier.fit(x_train,y_train)
roc_auc = tpot_classifier.score(x_test, y_test)
print(roc_auc)