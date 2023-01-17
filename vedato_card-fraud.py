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
!pip install ycimpute



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder

from sklearn import datasets, metrics, model_selection, svm

import missingno as msno

from ycimpute.imputer import iterforest,EM

from fancyimpute import KNN

from sklearn.preprocessing import OrdinalEncoder



import numpy as np

import pandas as pd 

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier



from warnings import filterwarnings

filterwarnings('ignore')



pd.set_option('display.max_columns', None)

import gc
%%time

# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

# WARNING! THIS CAN DAMAGE THE DATA 

def reduce_mem_usage2(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
encoder=OrdinalEncoder()

imputer=KNN()



def encode(data):

    '''function to encode non-null data and replace it in the original data'''

    #retains only non-null values

    nonulls = np.array(data.dropna())

    #reshapes the data for encoding

    impute_reshape = nonulls.reshape(-1,1)

    #encode date

    impute_ordinal = encoder.fit_transform(impute_reshape)

    #Assign back encoded values to non-null values

    data.loc[data.notnull()] = np.squeeze(impute_ordinal)

    return data
Ktest_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

Ktest_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")

Ktrain_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")

Ktrain_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
Ktrain= pd.merge(Ktrain_transaction, Ktrain_identity, on='TransactionID', how='left', left_index=True, right_index=True)

Ktest= pd.merge(Ktest_transaction, Ktest_identity, on='TransactionID', how='left', left_index=True, right_index=True)

#print(Ktrain.shape, Ktest.shape)

#print(Ktest.info())

#print(Ktrain.info())
Ktrain_cat=Ktrain.select_dtypes(include='object')

Ktest_cat =Ktest.select_dtypes(include='object')
#Ktrain_cat.nunique()
#Ktest_cat.nunique()
Ktrain_cat1=Ktrain_cat.drop(['P_emaildomain','R_emaildomain','id_30','id_31','id_33','DeviceInfo'], axis=1)

Ktest_cat1=Ktest_cat.drop(['P_emaildomain','R_emaildomain','id-30','id-31','id-33','DeviceInfo'], axis=1)
for i in Ktrain_cat1:

    encode(Ktrain[i])

for i in Ktest_cat1:

    encode(Ktest[i])
Ktrain_cat2=pd.concat([Ktrain['P_emaildomain'],Ktrain['R_emaildomain'],Ktrain['id_30'],Ktrain['id_31'],Ktrain['id_33'],Ktrain['DeviceInfo']], axis=1)

Ktest_cat2=pd.concat([Ktest['P_emaildomain'],Ktest['R_emaildomain'],Ktest['id-30'],Ktest['id-31'],Ktest['id-33'],Ktest['DeviceInfo']], axis=1)

for i in Ktrain_cat2:

    encode(Ktrain[i])

for i in Ktest_cat2:

    encode(Ktest[i])
import gc

del Ktest_identity

del Ktest_transaction

del Ktrain_identity

del Ktrain_transaction

del Ktrain_cat1

del Ktest_cat1

del Ktrain_cat2

del Ktest_cat2

gc.collect()
Ktest = reduce_mem_usage2(Ktest)

Ktrain = reduce_mem_usage2(Ktrain)
Ktrain.shape
Ktest.shape
Ktest.head()
z= Ktest.loc[:,'id-01':'id-38'].columns.str.replace('-','_')
z=list(z)

z
for x,y in zip(Ktest.loc[:,'id-01':'id-38'].columns, z):

    Ktest[y]=Ktest[x]

    del Ktest[x]

gc.collect()
y=Ktrain["isFraud"]

X=Ktrain.drop(["isFraud", "TransactionID"], axis=1).astype('float64')

X= X.fillna(-999)



Ktest_id = Ktest['TransactionID']

X_Ktest = Ktest.drop(['TransactionID'], axis=1).astype('float64')

X_Ktest = X_Ktest.fillna(-999)



X_Ktest = X_Ktest[X.columns]
# Ktest= Ktest[Ktrain.columns]
'''

X_fit=StandardScaler().fit_transform(X)

X_pca=PCA().fit(X_fit)

plt.plot(np.cumsum(X_pca.explained_variance_ratio_))

plt.title('All columns included', color='gray')

plt.xlabel("Number of Component", color='green')

plt.ylabel("Cumulative Variance Ratio", color='green')

plt.grid(color='gray', linestyle='-', linewidth=0.3)

plt.show()



X_Ktest_fit=StandardScaler().fit_transform(X_Ktest)

X_Ktest_pca=PCA().fit(X_Ktest_fit)

plt.plot(np.cumsum(X_Ktest_pca.explained_variance_ratio_))

plt.title('All columns included', color='gray')

plt.xlabel("Number of Component", color='green')

plt.ylabel("Cumulative Variance Ratio", color='green')

plt.grid(color='gray', linestyle='-', linewidth=0.3)

plt.show()

'''
# PCA Analysis is misleading here because we filled missing data. 



"""

#Final Model for Ktrain

X_pca = PCA(n_components=100).fit(X_fit)

X_fit = X_pca.fit_transform(X_fit)

X_pca.explained_variance_ratio_



#Final Model for Ktest



X_Ktest_pca = PCA(n_components=100).fit(X_Ktest_fit)

X_Ktest_fit = X_Ktest_pca.fit_transform(X_Ktest_fit)

print(X_pca.explained_variance_ratio_.sum())

print(X_Ktest_pca.explained_variance_ratio_.sum())

"""
'''

xlist=[]

for i in range(1,101):

    xlist.append("a"+str(i))



X_new= pd.DataFrame(data=X_fit, columns= xlist)

X_Ktest_new= pd.DataFrame(data=X_Ktest_fit, columns= xlist)

X_Ktest_new.head()

'''
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.25, random_state=42)
X = reduce_mem_usage2(X)

X_Ktest = reduce_mem_usage2(X_Ktest)
gc.collect()
print(Ktrain.shape)

print(Ktest.shape)

print(X.shape)

print(X_Ktest.shape)

# It gets so much time to run with each models, so passed for now.



'''models = [LogisticRegression,

          KNeighborsClassifier,

          GaussianNB,

          SVC,

          DecisionTreeClassifier,

          RandomForestClassifier,

          GradientBoostingClassifier,

          LGBMClassifier,

          XGBClassifier

          #CatBoostClassifier

         ]



def compML (df, y, algorithm):

    

    #y=df[y]

    #X=df.drop(["PassengerId","Survived"], axis=1).astype('float64')

    #X_train, X_test,y_train,y_test=train_test_split(X,y, test_size=0.25, random_state=42)

    

    model=algorithm().fit(X_train, y_train)

    y_pred=model.predict(X_test)

    accuracy= accuracy_score(y_test, y_pred)

    #return accuracy

    model_name= algorithm.__name__

    print(model_name,": ",accuracy)

    

    

for i in models:

    compML(X,"isFraud",i)

    

'''
model= LGBMClassifier().fit(X_train, y_train)

y_pred=model.predict(X_test)

accuracy_score(y_test,y_pred)
model
# Ktest_pred= lgb_model.predict(X_Ktest_new)
predictions=model.predict(X_Ktest)

output=pd.DataFrame({"TransactionID":Ktest_id, "isFraud":predictions})

output.to_csv('submission_model.csv', index=False)