# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/dataset2147b1d/train_file.csv')
train.head()
train.drop(columns=['Description','GeoLocation','QuestionCode','Patient_ID','YEAR'],inplace=True)
train.head()
test=pd.read_csv('../input/dataset2147b1d/test_file.csv')
test_pid=test.Patient_ID
train.head()
import category_encoders as ce
ce1=ce.TargetEncoder(cols = ['LocationDesc','Greater_Risk_Question','Race','Grade','StratID1','StratID2','StratID3'], min_samples_leaf = 20)
train.loc[:,['LocationDesc','Greater_Risk_Question','Race','Grade','StratID1','StratID2','StratID3']]=ce1.fit_transform(train.loc[:,['LocationDesc','Greater_Risk_Question','Race','Grade','StratID1','StratID2','StratID3']],train.loc[:,['Greater_Risk_Probability']])
train.head()
train=pd.get_dummies(data=train,columns=['Sex','StratificationType'])
train.head()
X=train.drop(columns=['Greater_Risk_Probability'])

Y=train['Greater_Risk_Probability']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=10,test_size=0.2)
from xgboost import XGBRegressor

#xgbr=XGBRegressor()
xgbr = XGBRegressor(colsample_bytree=0.4,

                    objective="reg:linear",

                 gamma=0.5,                 

                 learning_rate=0.01,

                 max_depth=5,

                 min_child_weight=1.5,

                 n_estimators=5000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42) 
from keras.models import Sequential

from keras.layers import Dense,Dropout,BatchNormalization
from catboost import CatBoostRegressor

sgd=CatBoostRegressor(iterations=10000,learning_rate=0.001,depth=5)
X_train.shape
xgbr.fit(X_train,Y_train)

#sgd.fit(X_train,Y_train)
xgbr.feature_importances_
from sklearn.metrics import mean_absolute_error
y_pred=xgbr.predict(X_test)
mean_absolute_error(Y_test,y_pred)
test.head()
test.drop(columns=['Description','GeoLocation','QuestionCode','Patient_ID','YEAR'],inplace=True)
test.head()
test.loc[:,['LocationDesc','Greater_Risk_Question','Race','Grade','StratID1','StratID2','StratID3']]=ce1.transform(test.loc[:,['LocationDesc','Greater_Risk_Question','Race','Grade','StratID1','StratID2','StratID3']])
test.head()
test=pd.get_dummies(data=test,columns=['Sex','StratificationType'])
test.head()
test_pred=xgbr.predict(test)
output=pd.DataFrame([np.array(test_pid).astype(np.int64),test_pred])
output=output.T
output.columns=['Patient_ID','Greater_Risk_Probability']
output.Greater_Risk_Probability=output.Greater_Risk_Probability.astype(np.float64)
output.head()
output.to_csv('submissionapr28v1.csv',index=None)
#tg=train.groupby('Greater_Risk_Question')
#train.Greater_Risk_Question.value_counts().plot(kind='bar')
#from matplotlib import pyplot as plt
#list(tg.groups.keys())
'''plt.suptitle('Histogram of Numerical Column', fontsize=20)

for i in range(1,21):

    plt.figure(num=20,figsize=(10,150))

    plt.subplot(20,2,i)

    f=plt.gca()

    f.set_title(list(tg.groups.keys())[i-1])

   # vals=np.size(train.iloc[tg.groups[list(tg.groups.keys())[i-1]],:]['Greater_Risk_Probability'].unique())

    plt.hist(train.iloc[tg.groups[list(tg.groups.keys())[i-1]],:]['Greater_Risk_Probability'],color='#3f5d7d')'''
#train.Sex.v
#test.Sex.value_counts()
#pk=train.groupby(['Greater_Risk_Question','Sex','Race']).Greater_Risk_Probability.median()
#len(pk)
#train.groupby(['LocationDesc',]).Greater_Risk_Probability.median().sort_values(ascending=True).plot(kind= 'bar',figsize=(50,10),fontsize=20,)
#train.iloc[tg.groups['Currently used marijuana'],:]['Greater_Risk_Probability'].hist(),

#plt.hist(train.iloc[tg.groups['Ever drank alcohol'],:]['Greater_Risk_Probability'])

#plt.show()
#train.iloc[tg.groups['Currently used marijuana'],:]['Grade'].value_counts().plot(kind='bar')
#train.iloc[tg.groups['Currently used marijuana'],:]['QuestionCode'].value_counts().plot(kind='bar')
#train.Greater_Risk_Question.value_counts().plot(kind='bar')
#train.StratificationType.value_counts()
#train.GeoLocation.value_counts().plot(kind='bar', figsize=(120,30),fontsize=40)
#train.groupby('LocationDesc').get_group('Houston, TX').GeoLocation.value_counts()
#train.GeoLocation = train.groupby(['LocationDesc'])['GeoLocation']\.transform(lambda x: x.fillna(x))
#train.isnull().sum()
#train[train.GeoLocation.isnull()].LocationDesc.value_counts()
#train[~train.GeoLocation.isnull()].LocationDesc.value_counts()
'''from geopy.geocoders import Nominatim



geolocator = Nominatim() 



for location in ('California USA', 'United States','Shelby County, TN'):

    geoloc = geolocator.geocode(location)

    print(location, ':', geoloc,(geoloc.latitude, geoloc.longitude))'''
mylist=[1,2,3]
import numpy as np

myarray = np.asarray(mylist)
myarray