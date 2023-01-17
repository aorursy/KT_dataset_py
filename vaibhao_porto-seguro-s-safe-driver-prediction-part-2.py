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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt, seaborn as sns

%matplotlib inline

from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report

from sklearn.utils import shuffle

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,StratifiedKFold

#getting the Data

data = pd.read_csv('../input/train.csv')
data.head()
#Getting the column information

y = data['target']

X = data.drop(['target','id'],axis= 1)

#Spliting the columns between categories 

catColumns = [col for col in X.columns if '_cat' in col ]

binColumns = [col for col in X.columns if '_bin' in col ]

contiColumns = [col for col in X.columns if ('_bin' not in col) and ('_cat' not in col) ]
#Evaluating the Categorical colums for missign data

plt.figure(figsize=(14,8))

sns.heatmap(X[catColumns].corr(),annot=True,cmap='coolwarm')



#getting colums with missing value information

for col in catColumns:

    c = X[col][X[col]== -1].count()

    if c > 0:

        print ('column -  {} has {} missing values. {:.2%}'.format(col,c,(c*1.0/len(X[col]))))



'''from sklearn.preprocessing import Imputer

imp = Imputer(missing_values=-1,strategy='most_frequent',axis=0)

'''

for col in catColumns:

    X[col].replace(to_replace = -1, value = X[col].std(), inplace = True)

        

#rechecking for any missing values

for col in catColumns:

    c = X[col][X[col]== -1].count()

    if c > 0 :

        print ('found missing column: {}'.format(col) )
#verifying missing values in Binary columns



for col in binColumns:

    c = X[col][X[col]== -1].count()

    if c > 0:

        print ('column -  {} has {} missing values. {:.2%}'.format(col,c,(c*1.0/len(X[col]))))
#Evaluating the Continuous colums for missign data

#filling all the missing column data with std



plt.figure(figsize=(20,14))

sns.heatmap(X[contiColumns].corr(),annot=True,cmap='coolwarm')



#getting colums with missing value information

for col in contiColumns:

    c = X[col][X[col]== -1].count()

    if c > 0:

        print ('column - {} has {} missing values. {:.2%}'.format(col,c,(c*1.0/len(X[col]))))

        X[col].replace(to_replace = -1, value = X[col].std(), inplace = True)



print ( 'All Mising values are updated with Std')



'''from sklearn.preprocessing import Imputer

imp = Imputer(missing_values=-1,strategy='most_frequent',axis=1)

'''

for col in catColumns:

    X[col].replace(to_replace = -1, value = X[col].std(), inplace = True)



#rechecking for any missing values

for col in catColumns:

    c = X[col][X[col]== -1].count()

    if c > 0 :

        print ('found missing column: {}'.format(col) )

    

print('no Missing columns')    
#verification of missing value in Dataframe

for col in X.columns:

    c = X[col][X[col]== -1].count()

    if c > 0 :

        print ('found missing column: {}'.format(col)) 

    

print('no Missing columns') 
X.head()
X = X.apply(lambda x : (x-x.min())/(x.max()-x.min()))
X.head()
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,StratifiedKFold

from sklearn.metrics import auc,roc_auc_score
#setting up the parameter for XGboost

fold = 4

paramCombinations = 4

param = {

    'min_child_weight' : [3,5,10,12],

    'gamma': [ 0.5,  1.5, 2],

    'subsample': [0.5, 0.7, 1.0],

    'colsample_bytree' : [0.6, 0.8, 1.0,0.3],

    'max_depth' : [3, 4, 5, 8, 10, 12],

    'learning_rate':[0.1, 0.01, 1, 0.5, 0.75, 0.03, 0.005]

}
skf = StratifiedKFold(random_state=42,shuffle=True,n_splits=fold)

xgb = XGBClassifier(learning_rate=0.03,n_estimators=300,silent=True,nthread=1)
rand_search = RandomizedSearchCV(xgb,param_distributions=param,verbose= 3,scoring= 'roc_auc',random_state=42,n_iter=paramCombinations, cv =skf.split(X,y))
rand_search.fit(X,y)
rand_search.best_score_
testData = pd.read_csv('../input/test.csv')
testData.head()
idTest = testData['id']

xTest = testData.drop('id',axis=1)



#Evaluating the Categorical colums for missign data

plt.figure(figsize=(14,10))

sns.heatmap(xTest[catColumns].corr(),annot=True,cmap='coolwarm')



#getting colums with missing value information

for col in catColumns:

    c = xTest[col][xTest[col]== -1].count()

    if c > 0:

        print ('column -  {} has {} missing values. {:.2%}'.format(col,c,(c*1.0/len(xTest[col]))))



'''from sklearn.preprocessing import Imputer

imp = Imputer(missing_values=-1,strategy='most_frequent',axis=0)

'''

for col in catColumns:

    xTest[col].replace(to_replace = -1, value = xTest[col].std(), inplace = True)

        

#rechecking for any missing values

for col in catColumns:

    c = xTest[col][xTest[col]== -1].count()

    if c > 0 :

        print ('found missing column: {}'.format(col) )
#verifying missing values in Binary columns

for col in binColumns:

    c = xTest[col][xTest[col]== -1].count()

    if c > 0:

        print ('column -  {} has {} missing values. {:.2%}'.format(col,c,(c*1.0/len(xTest[col]))))
#Evaluating the Continuous colums for missign data

#filling all the missing column data with std



plt.figure(figsize=(20,14))

sns.heatmap(X[contiColumns].corr(),annot=True,cmap='coolwarm')



#getting colums with missing value information

for col in contiColumns:

    c = xTest[col][xTest[col]== -1].count()

    if c > 0:

        print ('column - {} has {} missing values. {:.2%}'.format(col,c,(c*1.0/len(xTest[col]))))

        xTest[col].replace(to_replace = -1, value = xTest[col].std(), inplace = True)



print ( 'All Mising values are updated with Std')



'''from sklearn.preprocessing import Imputer

imp = Imputer(missing_values=-1,strategy='most_frequent',axis=1)

'''

for col in catColumns:

    xTest[col].replace(to_replace = -1, value = xTest[col].std(), inplace = True)



#rechecking for any missing values

for col in catColumns:

    c = xTest[col][xTest[col]== -1].count()

    if c > 0 :

        print ('found missing column: {}'.format(col) )

    

print('no Missing columns')    
#verification of missing value in Dataframe

for col in xTest.columns:

    c = xTest[col][xTest[col]== -1].count()

    if c > 0 :

        print ('found missing column: {}'.format(col)) 

    

print('no Missing columns') 
#Rescaling the Test data

xTest = xTest.apply(lambda x : (x-x.min())/(x.max()-x.min()))
pred = pd.DataFrame(rand_search.best_estimator_.predict_proba(xTest), columns=['1','target'])
pred.head()
PortoFile = pd.concat([idTest,pred['target']],axis=1)
PortoFile.tail()
#Exporting the file

PortoFile.to_csv('Porto1.csv',index=False)