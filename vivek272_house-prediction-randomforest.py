# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt 

import seaborn as sns 

from sklearn.ensemble import RandomForestRegressor



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
file_path="../input/house-prices-advanced-regression-techniques/train.csv"

file_test="../input/house-prices-advanced-regression-techniques/test.csv"

data=pd.read_csv(file_path)

test=pd.read_csv(file_test)

#set(data['Street'])

data.head()

y=data.SalePrice

#print(y)

features=["MSSubClass","MSZoning","LotFrontage","LotArea","LotShape","LotConfig","Neighborhood","Condition1","Condition2","BldgType","OverallQual","YearRemodAdd","CentralAir","Electrical","1stFlrSF",

          "2ndFlrSF","GrLivArea","BedroomAbvGr","KitchenAbvGr","Functional","Fireplaces","SaleType","SaleCondition"]

#Creating X

X=data[features]

X2=test[features]

#print(X.describe())

#print(X.isnull())

#print(X.mean())

X=X.fillna(X.mean())

X2=X2.fillna(X2.mean())

X2.head()
set(X['Neighborhood'])
X['Neighborhood'] = [1 if i == 'Blmngtn' else 2 if i=='Blueste' else 3 if i=='BrDale' else 4 if i =='BrkSide' else 5 if i =='ClearCr' else 6 if i =='CollgCr' else 7 if i =='Crawfor' else 8 if i =='Edwards' else 9 if i =='Gilbert' else 10 if i =='IDOTRR' else 11 if i =='MeadowV' else 12 if i =='Mitchel' else 13 if i =='NAmes' else 14 if i =='NPkVill' else 15 if i =='NWAmes' else 16 if i =='NoRidge' else 17 if i =='NridgHt' else 18 if i =='OldTown' else 19 if i =='SWISU' else 20 if i =='Sawyer' else 21 if i =='SawyerW' else 22 if i =='Somerst' else 23 if i =='StoneBr' else 24 if i =='Timber' else 25 for i in X['Neighborhood'] ]



X2['Neighborhood'] = [1 if i == 'Blmngtn' else 2 if i=='Blueste' else 3 if i=='BrDale' else 4 if i =='BrkSide' else 5 if i =='ClearCr' else 6 if i =='CollgCr' else 7 if i =='Crawfor' else 8 if i =='Edwards' else 9 if i =='Gilbert' else 10 if i =='IDOTRR' else 11 if i =='MeadowV' else 12 if i =='Mitchel' else 13 if i =='NAmes' else 14 if i =='NPkVill' else 15 if i =='NWAmes' else 16 if i =='NoRidge' else 17 if i =='NridgHt' else 18 if i =='OldTown' else 19 if i =='SWISU' else 20 if i =='Sawyer' else 21 if i =='SawyerW' else 22 if i =='Somerst' else 23 if i =='StoneBr' else 24 if i =='Timber' else 25 for i in X2['Neighborhood'] ]

set(X['Neighborhood'])
set(X['MSZoning'])
X2['MSZoning'] = [1 if i == 'C (all)' else 2 if i=='FV' else 3 if i=='RH' else 4 if i =='RL' else 5 if i =='RM' else 6 for i in  X2['MSZoning'] ]

X['MSZoning'] = [1 if i == 'C (all)' else 2 if i=='FV' else 3 if i=='RH' else 4 if i =='RL' else 5 if i =='RM' else 6 for i in  X['MSZoning'] ]





set(X['LotShape'])
X['LotShape'] = [1 if i == 'IR1' else 2 if i=='IR2' else 3 if i=='IR3' else 4 if i =='Reg' else 5 for i in  X['LotShape'] ]

X2['LotShape'] = [1 if i == 'IR1' else 2 if i=='IR2' else 3 if i=='IR3' else 4 if i =='Reg' else 5 for i in  X2['LotShape'] ]
set(X['LotConfig'])
X['LotConfig'] = [1 if i == 'Corner' else 2 if i=='CulDSac' else 3 if i=='FR2' else 4 if i =='FR3' else 5 if i =='Inside' else 6 for i in  X['LotConfig'] ]

X2['LotConfig'] = [1 if i == 'Corner' else 2 if i=='CulDSac' else 3 if i=='FR2' else 4 if i =='FR3' else 5 if i =='Inside' else 6 for i in  X2['LotConfig'] ]

set(X['Condition1'])
X['Condition1']=[1 if i == 'Artery' else 2 if i=='Feedr' else 3 if i=='Norm' else 4 if i =='PosA' else 5 if i =='RRAe' else 

                 6 if i =='RRAn' else 7 if i =='RRNe' else 8 if i=='RRNn' else 9 if i=='PosN' else 10  for i in X['Condition1']]

X2['Condition1']=[1 if i == 'Artery' else 2 if i=='Feedr' else 3 if i=='Norm' else 4 if i =='PosA' else 5 if i =='RRAe' else 

                 6 if i =='RRAn' else 7 if i =='RRNe' else 8 if i=='RRNn' else 9 if i=='PosN' else 10  for i in X2['Condition1']]



set(X['Condition2'])
X['Condition2']=[1 if i == 'Artery' else 2 if i=='Feedr' else 3 if i=='Norm' else 4 if i =='PosA' else 5 if i =='RRAe' else 

                 6 if i =='RRAn' else 7 if i =='RRNn' else 8 if i=='PosN' else 9  for i in X['Condition2']]

X2['Condition2']=[1 if i == 'Artery' else 2 if i=='Feedr' else 3 if i=='Norm' else 4 if i =='PosA' else 5 if i =='RRAe' else 

                 6 if i =='RRAn' else 7 if i =='RRNn' else 8 if i=='PosN' else 9  for i in X2['Condition2']]

set(X['Electrical'])
X['Electrical']=[1 if i == 'FuseA' else 2 if i=='FuseF' else 3 if i=='FuseP' else 4 if i =='Mix' else 5 if i =='SBrkr' else 

                 6 if i =='nan' else 7  for i in X['Electrical']]

X2['Electrical']=[1 if i == 'FuseA' else 2 if i=='FuseF' else 3 if i=='FuseP' else 4 if i =='Mix' else 5 if i =='SBrkr' else 

                 6 if i =='nan' else 7  for i in X2['Electrical']]



set(X['BldgType'])
X['BldgType'] = [1 if i == '1Fam' else 2 if i=='2fmCon' else 3 if i=='Duplex' else 4 if i =='Twnhs' else 5  for i in X['BldgType'] ]



X2['BldgType'] = [1 if i == '1Fam' else 2 if i=='2fmCon' else 3 if i=='Duplex' else 4 if i =='Twnhs' else 5  for i in X2['BldgType'] ]
set(X['CentralAir'])
X['CentralAir']=[1 if i=='Y' else 2 for i in X['CentralAir']]

X2['CentralAir']=[1 if i=='Y' else 2 for i in X2['CentralAir']]
set(X['Functional'])
X['Functional']=[1 if i == 'Maj1' else 2 if i=='Maj2' else 3 if i=='Min1' else 4 if i =='Min2' else 5 if i =='Mod' else 

                 6 if i =='Sev' else 7 if i =='Typ' else 8  for i in X['Functional']]



X2['Functional']=[1 if i == 'Maj1' else 2 if i=='Maj2' else 3 if i=='Min1' else 4 if i =='Min2' else 5 if i =='Mod' else 

                 6 if i =='Sev' else 7 if i =='Typ' else 8  for i in X2['Functional']]

set(X['SaleType'])
X['SaleType']=[1 if i == 'COD' else 2 if i=='CWD' else 3 if i=='Con' else 4 if i =='ConLD' else 5 if i =='ConLI' else 

                 6 if i =='ConLw' else 7 if i =='New' else 8 if i=="Oth" else 9 if i=="WD" else 10  for i in X['SaleType']]

X2['SaleType']=[1 if i == 'COD' else 2 if i=='CWD' else 3 if i=='Con' else 4 if i =='ConLD' else 5 if i =='ConLI' else 

                 6 if i =='ConLw' else 7 if i =='New' else 8 if i=="Oth" else 9 if i=="WD" else 10  for i in X2['SaleType']]

set(X['SaleCondition'])
X['SaleCondition']=[1 if i == 'Abnorml' else 2 if i=='AdjLand'

                    else 3 if i=='Alloca' else 4 if i =='Family' 

                    else 5 if i=='Normal' else 6 if i=='Partial' else 7  for i in X['BldgType'] ]

X2['SaleCondition']=[1 if i == 'Abnorml' else 2 if i=='AdjLand'

                    else 3 if i=='Alloca' else 4 if i =='Family' 

                    else 5 if i=='Normal' else 6 if i=='Partial' else 7  for i in X2['BldgType'] ]
X2
plt.figure(figsize=(15,8))

sns.distplot(X['LotFrontage'])

plt.show()
X=X.fillna(X.mean())

#X

X2=X2.fillna(X2.mean())

X2
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = RandomForestRegressor(random_state=1)

model.fit(train_X,train_y)
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=1)

    model.fit(train_X,train_y)

    preds_vals=model.predict(val_X)

    mae=mean_absolute_error(preds_vals,val_y)

    return(mae)
val_predictions = model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)


#print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

candidate_max_leaf_nodes = [5,38,90,200,200,500]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for  max_leaf_nodes in [5,38,90,200,200,500]:

    my_mae=get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

best_tree_size=38
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

final_model.fit(X, y)

final_model
res=final_model.predict(X2)
file_path3="../input/house-prices-advanced-regression-techniques/sample_submission.csv"

sub =pd.read_csv(file_path3)
sub['SalesPrice']=res

sub
sub.drop(columns="SalePrice",axis=1)

#sub.columns=['Id','SalePrice']
sub.to_csv('fourth_sub')
sub