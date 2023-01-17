# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df.head()
print(df['MSZoning'].value_counts())



sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
df.info()




df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())



df.drop(['Alley'],axis=1,inplace=True)



df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])

df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])



df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])

df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])



df.drop(['GarageYrBlt'],axis=1,inplace=True)



df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])

df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])

df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])



df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
df.drop(['Id'],axis=1,inplace=True)



df.isnull().sum()
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])

df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])



sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])



sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df.dropna(inplace=True)
df.shape
df_cat = df.select_dtypes('object')

df_cont = df.select_dtypes(['int64','float64'])
plt.figure(figsize=(15,5))

for i in range(0,5):

    plt.subplot(1, 5, i+1)

    plt.xticks(rotation=0)

    sns.distplot(df[df_cont.columns[i]])
plt.figure(figsize=(15,5))

for i in range(0,5):

    plt.subplot(1, 5, i+1)

    plt.xticks(rotation=0)

    sns.boxplot(df[df_cont.columns[i]])
plt.figure(figsize=(15,5))

for i in range(5,9):

    plt.subplot(1, 5, i-4)

    plt.xticks(rotation=0)

    sns.distplot(df[df_cont.columns[i]])
plt.figure(figsize=(15,5))

for i in range(5,9):

    plt.subplot(1, 5, i-4)

    plt.xticks(rotation=0)

    sns.boxplot(df[df_cont.columns[i]])
sns.pairplot(df[['MasVnrArea','SalePrice']]);
plt.figure(figsize=(40,20))

sns.heatmap(df_cont.corr(),annot=True,cmap="YlGnBu",linewidths=0.5);
plt.figure(figsize=(15,5))

for i in range(0,5):

    plt.subplot(1, 5, i+1)

    plt.title(df_cat.columns[i])

    sns.countplot(y=df_cat.columns[i], data=df_cat)
plt.figure(figsize=(15,5))

for i in range(5,10):

    plt.subplot(1, 5, i-4)

    plt.title(df_cat.columns[i])

    sns.countplot(y=df_cat.columns[i], data=df_cat)
plt.figure(figsize=(15,5))

for i in range(10,15):

    plt.subplot(1, 5, i-9)

    plt.title(df_cat.columns[i])

    sns.countplot(y=df_cat.columns[i], data=df_cat)
plt.figure(figsize=(15,5))

for i in range(15,20):

    plt.subplot(1, 5, i-14)

    plt.title(df_cat.columns[i])

    sns.countplot(y=df_cat.columns[i], data=df_cat)
plt.figure(figsize=(15,5))

for i in range(20,25):

    plt.subplot(1, 5, i-19)

    plt.title(df_cat.columns[i])

    sns.countplot(y=df_cat.columns[i], data=df_cat)
plt.figure(figsize=(15,5))

for i in range(25,30):

    plt.subplot(1, 5, i-24)

    plt.title(df_cat.columns[i])

    sns.countplot(y=df_cat.columns[i], data=df_cat)
plt.figure(figsize=(15,5))

for i in range(30,35):

    plt.subplot(1, 5, i-29)

    plt.title(df_cat.columns[i])

    sns.countplot(y=df_cat.columns[i], data=df_cat)
plt.figure(figsize=(15,5))

for i in range(35,38):

    plt.subplot(1, 5, i-34)

    plt.title(df_cat.columns[i])

    sns.countplot(y=df_cat.columns[i], data=df_cat)
plt.figure(figsize=(15,5))

for i in range(0,5):

    plt.subplot(1, 5, i+1)

    sns.boxplot(x=df[df_cat.columns[i]],y=df['SalePrice'],data=df)

    #plt.xscale('log')

    plt.xticks(rotation=90)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=3, hspace=None)
plt.figure(figsize=(15,5))

for i in range(5,10):

    plt.subplot(1, 5, i-4)

    sns.boxplot(x=df[df_cat.columns[i]],y=df['SalePrice'],data=df)

    #plt.xscale('log')

    plt.xticks(rotation=90)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=3, hspace=None)
sns.boxplot(df_cat['Neighborhood'],df['SalePrice']);

plt.xticks(rotation=90);
sns.boxplot(df_cat['BldgType'],df['SalePrice']);

plt.xticks(rotation=90);
sns.boxplot(df_cat['HouseStyle'],df['SalePrice']);

plt.xticks(rotation=90);
sns.boxplot(df_cat['RoofStyle'],df['SalePrice']);

plt.xticks(rotation=90);
sns.boxplot(df_cat['RoofMatl'],df['SalePrice']);

plt.xticks(rotation=90);
sns.boxplot(df_cat['Exterior1st'],df['SalePrice']);

plt.xticks(rotation=90);
plt.figure(figsize=(15,5))

for i in range(16,21):

    plt.subplot(1, 5, i-15)

    sns.boxplot(x=df[df_cat.columns[i]],y=df['SalePrice'],data=df)

    #plt.xscale('log')

    plt.xticks(rotation=90)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=3, hspace=None)
plt.figure(figsize=(15,5))

for i in range(21,26):

    plt.subplot(1, 5, i-20)

    sns.boxplot(x=df[df_cat.columns[i]],y=df['SalePrice'],data=df)

    #plt.xscale('log')

    plt.xticks(rotation=90)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=3, hspace=None)
plt.figure(figsize=(15,5))

for i in range(26,30):

    plt.subplot(1, 5, i-25)

    sns.boxplot(x=df[df_cat.columns[i]],y=df['SalePrice'],data=df)

    #plt.xscale('log')

    plt.xticks(rotation=90)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=3, hspace=None)
plt.figure(figsize=(15,5))

for i in range(30,35):

    plt.subplot(1, 5, i-29)

    sns.boxplot(x=df[df_cat.columns[i]],y=df['SalePrice'],data=df)

    #plt.xscale('log')

    plt.xticks(rotation=90)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=3, hspace=None)
plt.figure(figsize=(20,5))

for i in range(35,38):

    plt.subplot(1, 4, i-34)

    sns.boxplot(x=df[df_cat.columns[i]],y=df['SalePrice'],data=df)

    #plt.xscale('log')

    plt.xticks(rotation=90)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=3, hspace=None)
plt.figure(figsize=(10,5))

sns.boxplot(x = df_cat['SaleType'],y=df['SalePrice'],hue=df_cat['SaleCondition']);
plt.figure(figsize=(10,5))

sns.boxplot(x = df_cat['PavedDrive'],y=df['SalePrice'],hue=df_cat['SaleCondition']);
plt.figure(figsize=(10,5))

sns.boxplot(x = df_cat['PavedDrive'],y=df['SalePrice'],hue=df_cat['SaleType']);
plt.figure(figsize=(10,5))

sns.boxplot(x = df_cat['GarageFinish'],y=df['SalePrice'],hue=df_cat['GarageType']);
plt.figure(figsize=(10,5))

sns.boxplot(x = df_cat['GarageQual'],y=df['SalePrice'],hue=df_cat['GarageType']);
plt.figure(figsize=(10,5))

sns.boxplot(x = df_cat['Electrical'],y=df['SalePrice'],hue=df_cat['KitchenQual']);
plt.figure(figsize=(10,5))

sns.boxplot(x = df_cat['Heating'],y=df['SalePrice'],hue=df_cat['HeatingQC']);
plt.figure(figsize=(10,5))

sns.boxplot(x = df_cat['RoofMatl'],y=df['SalePrice'],hue=df_cat['RoofStyle']);
plt.figure(figsize=(10,5))

sns.boxplot(x = df_cat['HouseStyle'],y=df['SalePrice'],hue=df_cat['BldgType']);
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',

         'Condition2','BldgType','Condition1','HouseStyle','SaleType',

        'SaleCondition','ExterCond',

         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',

         'CentralAir',

         'Electrical','KitchenQual','Functional',

         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
def category_onehot_multcols(multcolumns):

    df_final=final_df

    i=0

    for fields in multcolumns:

        

        print(fields)

        df1=pd.get_dummies(final_df[fields],drop_first=True)

        

        final_df.drop([fields],axis=1,inplace=True)

        if i==0:

            df_final=df1.copy()

        else:

            

            df_final=pd.concat([df_final,df1],axis=1)

        i=i+1

       

        

    df_final=pd.concat([final_df,df_final],axis=1)

        

    return df_final
main_df=df.copy()
test_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_df.shape
#check null values

test_df.isnull().sum()
## Fill Missing Values



test_df['LotFrontage']=test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())



test_df['MSZoning']=test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])
test_df.drop(['Alley'],axis=1,inplace=True)
test_df['BsmtCond']=test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])

test_df['BsmtQual']=test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])



test_df['FireplaceQu']=test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])

test_df['GarageType']=test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])



test_df.drop(['GarageYrBlt'],axis=1,inplace=True)

test_df.shape
test_df['GarageFinish']=test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])

test_df['GarageQual']=test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])

test_df['GarageCond']=test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])



test_df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
test_df.drop(['Id'],axis=1,inplace=True)



test_df['MasVnrType']=test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])

test_df['MasVnrArea']=test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mode()[0])
sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test_df['BsmtExposure']=test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0])



sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test_df['BsmtFinType2']=test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0])
test_df['Utilities']=test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])

test_df['Exterior1st']=test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])

test_df['Exterior2nd']=test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])

test_df['BsmtFinType1']=test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0])

test_df['BsmtFinSF1']=test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean())

test_df['BsmtFinSF2']=test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean())

test_df['BsmtUnfSF']=test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean())

test_df['TotalBsmtSF']=test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean())

test_df['BsmtFullBath']=test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mode()[0])

test_df['BsmtHalfBath']=test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mode()[0])

test_df['KitchenQual']=test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])

test_df['Functional']=test_df['Functional'].fillna(test_df['Functional'].mode()[0])

test_df['GarageCars']=test_df['GarageCars'].fillna(test_df['GarageCars'].mean())

test_df['GarageArea']=test_df['GarageArea'].fillna(test_df['GarageArea'].mean())

test_df['SaleType']=test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])

test_df.shape
final_df=pd.concat([df,test_df],axis=0)

final_df.shape
final_df=category_onehot_multcols(columns)
final_df.shape
final_df =final_df.loc[:,~final_df.columns.duplicated()]

final_df.shape
df_Train=final_df.iloc[:1422,:]

df_Test=final_df.iloc[1422:,:]
# all values are null

df_Test.drop(['SalePrice'],axis=1,inplace=True)
X_train=df_Train.drop(['SalePrice'],axis=1)

y_train=df_Train['SalePrice']
import xgboost

regressor=xgboost.XGBRegressor()
booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]
n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }
from sklearn.model_selection import RandomizedSearchCV

random_cv = RandomizedSearchCV(estimator=regressor,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)
random_cv.fit(X_train,y_train)
random_cv.best_estimator_
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,

       max_depth=2, min_child_weight=1, missing=None, n_estimators=900,

       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

       silent=True, subsample=1)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(df_Test)

pred=pd.DataFrame(y_pred)

sub_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets=pd.concat([sub_df['Id'],pred],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('sample_submission.csv',index=False)
import pickle

filename = 'finalized_xgboost_model.pkl'

pickle.dump(regressor, open(filename, 'wb'))
pred.columns=['SalePrice']

temp_df=df_Train['SalePrice'].copy()

temp_df.column=['SalePrice']

df_Train.drop(['SalePrice'],axis=1,inplace=True)
df_Train=pd.concat([df_Train,temp_df],axis=1)
df_Test=pd.concat([df_Test,pred],axis=1)
df_Train=pd.concat([df_Train,df_Test],axis=0)
X_train=df_Train.drop(['SalePrice'],axis=1)

y_train=df_Train['SalePrice']
# Importing the Keras libraries and packages

import tensorflow.keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LeakyReLU,PReLU,ELU

from tensorflow.keras.layers import Dropout

from tensorflow.keras import regularizers





model = Sequential()

model.add(Dense(200, input_dim=174,kernel_initializer = 'he_uniform', activation= "relu"))

model.add(Dense(100,kernel_initializer = 'he_uniform', activation= "relu"))

model.add(Dense(50,kernel_initializer = 'he_uniform', activation= "relu"))

model.add(Dense(1,kernel_initializer = 'he_uniform'))

model.summary()
model.compile(loss= "mean_squared_error" , optimizer="Adam", metrics=["mean_squared_error"])
# Fitting the ANN to the Training set

model_history=model.fit(X_train, y_train,validation_split=0.2, batch_size = 32, epochs = 500)
plt.plot(model_history.history['loss'], label='train')

plt.plot(model_history.history['val_loss'], label='test')

plt.legend()

plt.show()
y_train_pred = model.predict(X_train)
ann_pred=model.predict(df_Test.drop('SalePrice',axis=1))
pred=pd.DataFrame(ann_pred)

sub_df=pd.read_csv('sample_submission.csv')

datasets=pd.concat([sub_df['Id'],pred],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('sample_submission_ANN.csv',index=False)