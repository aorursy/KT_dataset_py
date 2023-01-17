# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
#uploading training and test data



import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.head()
#saving outcome in Sale_Price

Sale_Price=train.iloc[:,80]

Sale_Price.shape
train.shape
#here we checking data summury

train.describe()
#droping SalePrice column

train=train.drop(["SalePrice"],axis=1)

train.head()


test.head()
test.shape
#combining training & testing data for preposesing

data= pd.concat([train,test], keys=['x', 'y'])

data.head()
data.shape
#checking missing values

data.isnull().sum()
data.info()
#drroping unnessery column bcoz they have 80% + nan values



data=data.drop(["Id","Fence","MiscFeature","PoolQC","Alley"],axis=1)
#finding numeric column from data



num_col=data._get_numeric_data().columns.tolist()

num_col
#filling numrical missing value using fillna

for col in num_col:

    data[col].fillna(data[col].mean(),inplace=True)
#finding catogorical features

cat_col=set(data.columns)-set(num_col)

cat_col
# filling catgorical missing value



for col in cat_col:



    data[col].fillna(data[col].mode()[0],inplace=True)
data.info()
#count total value in every catgorical feature

for i in cat_col:

    print(data[i].value_counts())
data.shape
#droping some unnecessary cat_features bcoz they have 80% + same value and 20% - defertnt values so they can't effect score

data=data.drop(["GarageCond","RoofMatl","Heating","Condition2","BsmtCond","GarageQual","SaleType","CentralAir","Functional","FireplaceQu","Electrical","LandSlope","ExterCond","Condition1"],axis=1)
data.shape
import seaborn as sns



plt.figure(figsize=(20, 20))

sns.heatmap(train.corr(), annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")

plt.show()
#droping some unnecessary num_features by using heatmap

data=data.drop(["MSSubClass","OverallCond","BsmtFinSF2","KitchenAbvGr","BsmtHalfBath","LowQualFinSF","YrSold","MoSold","3SsnPorch","EnclosedPorch","PoolArea","ScreenPorch"],axis=1)
data.shape
#here we use one hot encoading to encoad cat_features



X=pd.get_dummies(data)

X.shape
#Training data after preproscing



Train_data=X.loc["x"]

Train_data.shape
#Testing data after preproscing

Test_data=X.loc["y"]

Test_data.shape
#here we use minmax scaler for scaling numeric fields

scalerX = MinMaxScaler(feature_range=(0, 1))

X[X.columns] = scalerX.fit(X[X.columns])
#here we can see traning data

Train_data.head()
#here we add salePrice column in traning data 

Train_data.insert(2,column="SalePrice",value=Sale_Price)

Train_data.head()
#here we split data in input(x) and output(y)

x=Train_data.drop(["SalePrice"],axis=True)

y=Train_data["SalePrice"]
#scaling data using minMax scaler

x[x.columns] = scalerX.transform(x[x.columns])
#spliting Training data for traning model and cheak score 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=40)
#here we use Random Forest Regressor for model building



from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(n_estimators = 50,random_state=40,min_impurity_decrease=0.002,min_weight_fraction_leaf=0.001,min_samples_split=5)

rfr.fit(x_train,y_train)

y_predictrfc = rfr.predict(x_test)



#here we can check our model score

print(rfr.score(x_test,y_test))
#here we use anthoar algo to findout best algo

from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor(random_state=140,min_samples_split=5,min_impurity_decrease=0.002,min_weight_fraction_leaf=0.001)

dtr.fit(x_train,y_train)



#u can also use GridSearchCV / random Searchcv for hyperperameter tuning

print(dtr.score(x_test,y_test))
#here we see test data here one column is missing that is Saleprice bcoz that is need to predict

Test_data.head()
Test_data.shape
#here we scaling texting data using MinMax scaler

Test_data[Test_data.columns] = scalerX.transform(Test_data[Test_data.columns])
#here we predict SalePrice using RFR model

y_model_prerfc = rfr.predict(Test_data)
#Here we can See predict Sale Price

y_model_prerfc=np.around(y_model_prerfc,0)

y_model_prerfc
