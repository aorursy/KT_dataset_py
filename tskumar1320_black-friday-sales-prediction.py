import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import RandomizedSearchCV
# !pip install lightgbm 
pd.set_option('display.max_rows', 5000000)
companyData=pd.read_csv("/kaggle/input/train.csv")

companytestData=pd.read_csv("/kaggle/input/test.csv")
companyData.head()
companytestData.head()
companyData.isnull().sum() 
companytestData.isnull().sum() 
copyData=companyData.copy()
#CHeck is there any patterns to fill category values '0' with other value

np.sort(copyData['Purchase'].unique())

# for i in np.sort(copyData['Purchase'].unique()):

#   print(copyData[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']][copyData['Purchase']==12])

#   break;



# dont see any patterns to fill NANs 
# so Impute Nan to 0 

copyData['Product_Category_2'].fillna(copyData['Product_Category_2'], axis=0, inplace=True)

copyData['Product_Category_3'].fillna(copyData['Product_Category_3'], axis=0, inplace=True)



companytestData['Product_Category_2'].fillna(companytestData['Product_Category_2'], axis=0, inplace=True)

companytestData['Product_Category_3'].fillna(companytestData['Product_Category_3'], axis=0, inplace=True)



#drop Purchase Nan row

copyData.dropna(axis=0, inplace=True)

companytestData.dropna(axis=0, inplace=True)
copyData.columns
#Drop unnecessary columns

copyData.drop(['User_ID','Product_ID'], axis=1, inplace=True)
copyData.columns
#Check Gender count

sns.countplot('Gender', data=copyData)

plt.show()
# Gender wise Age

sns.countplot('Gender', data=copyData, hue='Age')

plt.show()
# Maritual Status Count

sns.countplot('Marital_Status', data=copyData)

plt.show()
# Martitual Status by Age

sns.countplot('Marital_Status', data=copyData, hue='Age')

plt.show()
# Martitual Status by Age

sns.countplot('Occupation', data=copyData)

plt.show()
# Martitual Status by Age

sns.countplot('Occupation', data=copyData, hue='Age')

plt.show()
# Martitual Status by Age

sns.countplot('Occupation', data=copyData, hue='Product_Category_1')

plt.show()
#UNique Values

dict={}

def GetUniqueValues(df):

  for i in df:

    # print(df[i].unique())

    dict[i]=(df[i].unique())

  return dict



GetUniqueValues(copyData)

# GetUniqueValues(companytestData)
# Remove + from Stay_In_Current_City_Years

copyData['Stay_In_Current_City_Years']=(copyData['Stay_In_Current_City_Years'].str.strip('+').astype('float'))

companytestData['Stay_In_Current_City_Years']=(companytestData['Stay_In_Current_City_Years'].str.strip('+').astype('float'))
#Handle Categories

cat_column=[]

for i in copyData.columns:

  if (copyData[i].dtypes=='object'):

    cat_column.append(i)



print(cat_column)
companytestData1=companytestData.copy()
copyData=pd.get_dummies(copyData,  drop_first=True )

companytestData1=pd.get_dummies(companytestData1, columns=cat_column, drop_first=True )

companytestData1.head()
copyData.head()
# sns.pairplot(copyData, hue='Purchase')

# plt.show()
X=copyData.drop(['Purchase'], axis=1)

Y=(copyData['Purchase']).ravel()

print('X Shape', X.shape)

print('Y Shape', Y.shape)



X_Train, X_Test, Y_Train, Y_Test=train_test_split(X,Y, test_size=0.3, random_state=56)





print('X_Train Shape', X_Train.shape)

print('X_Test Shape', X_Test.shape)

print('Y_Train Shape', Y_Train.shape)

print('Y_Test Shape', Y_Test.shape)

gbm=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,

                          learning_rate=0.1, loss='ls', max_depth=3,

                          max_features=None, max_leaf_nodes=None,

                          min_impurity_decrease=0.0, min_impurity_split=None,

                          min_samples_leaf=1, min_samples_split=2,

                          min_weight_fraction_leaf=0.0, n_estimators=700,

                          n_iter_no_change=None, presort='auto',

                          random_state=None, subsample=1.0, tol=0.0001,

                          validation_fraction=0.1, verbose=0, warm_start=False)

gbm.fit(X_Train, Y_Train)

  
Pred_Y=gbm.predict(companytestData1.drop(['User_ID','Product_ID'], axis=1))
companytestData1['Purchase']=Pred_Y
companytestData1['Comb'] =companytestData1['User_ID'].astype(str) + companytestData1['Product_ID'].astype(str)
companytestData1[['Comb','Purchase']].to_csv(r'Sample_Submission.csv',index=False)
companytestData1.head()