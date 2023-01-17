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
train = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
test = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
train.head()
train=pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
test=pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')

train['Gender'] = train['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
test['Gender'] = test['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)

for column in ['Region_Code','Policy_Sales_Channel']:
    train[column] = train[column].astype('int')
    test[column] = test[column].astype('int')
    
id=test.id  # capture id for later, test id only, for final submission
    
train=train.drop('id',axis=1) # drop the id column
test=test.drop('id',axis=1)

cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
 'Region_Code', 'Policy_Sales_Channel']

for column in cat_feat:
    train[column] = train[column].astype('str')

for column in cat_feat:
    test[column] = test[column].astype('str')

train['Age'] = train['Age']//5 # divide all ages by 5 to get them into bins of 5 years each for the dummy variables
train['Age'] = train['Age'].astype(str)

test['Age'] = test['Age']//5
test['Age'] = test['Age'].astype(str)

train['Annual_Premium'] = ((train['Annual_Premium'])//1000)**0.5//1 #bin the annual premium into about 20 bins, with smaller bin sizes for smaller amounts
train['Annual_Premium'] = train['Annual_Premium'].astype(str)
test['Annual_Premium'] = ((test['Annual_Premium'])//1000)**0.5//1
test['Annual_Premium'] = test['Annual_Premium'].astype(str)

train=pd.get_dummies(train,drop_first=True)
test=pd.get_dummies(test,drop_first=True)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

mm = MinMaxScaler()
train[['Vintage']] = mm.fit_transform(train[['Vintage']])
test[['Vintage']] = mm.fit_transform(test[['Vintage']])  # This simply reduces Vintage to smaller numbers.  They are not expected to make a difference to the model because the distribution is flat

rejectColumns = []
i = 0
for name in list(train.columns):
    if name not in list(test.columns):
        #print(name)
        rejectColumns.append(name)
#print(i)
print(rejectColumns)

rejectColumns.remove('Response')

for name in train.columns:
    #print(name)
    if name in rejectColumns:
        #print(name)
        train = train.drop(name,axis = 1)
        
i = 0
rejectColumns = []
for name in list(test.columns):
    if name not in list(train.columns):
        #print(name)
        rejectColumns.append(name)
        i += 1
        
for name in test.columns:
    #print(name)
    if name in rejectColumns:
        print(name)
        test = test.drop(name,axis = 1)
        
from sklearn.model_selection import train_test_split
train_target=train['Response']
train=train.drop(['Response'], axis = 1)
x_train,x_test,y_train,y_test = train_test_split(train,train_target, random_state = 0)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
# Deep Learning Libraries
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
model = tf.keras.models.load_model('../input/tfmodel/tfModelInsurance.h5')
# This is a useful function for finding duplicates
'''len(train.drop(train.drop_duplicates().index)) # 50386
traindups = train.drop(train.drop_duplicates().index) # first round
traindups = traindups.drop(traindups.drop_duplicates().index) # 2nd round, 16447, 3rd round, 6166; 4th, 2305; 5th, 844; 6th, 287; 7th, 84; 8th, 16
len(traindups) # 20103'''
# Region code 28 is by far the most common region
# Policy sales channel 26 is by far the most common
# the combination is 37,459 times.  The combination of region 28 and policy sales 124 is 27,546
model.get_weights() #6 / 234[24, 24, 24 ...], 24, 24[20, 20 ...], 20, 20[2, 2 ...], 2: [ 0.02806583, -0.02806598]
model.get_weights()[2][0] # series of weights for multiplying the first hidden layer to calculate the second hidden layer.
layer1wtsum = []
for i in range(234):
    layer1wtsum.append(sum(model.get_weights()[0][i]))
layer1wtsumdf = pd.DataFrame({'column': train.columns, 'weightsum': layer1wtsum})
layer1wtsumdf[:20] # scattered values all over the place, no clear trend with age
layer1wtsumdf.sort_values(by='weightsum')[:15]
layer1wtsumdf.sort_values(by='weightsum')[-15:]
trainReg28Pol26 = train.loc[(train.Region_Code_28 == 1) & (train.Policy_Sales_Channel_26 == 1),:]
trainReg28Pol26['Vintage'].describe() # Confirms that the range goes from 0 to 1.  The default will be set to 0.5
exampledf = trainReg28Pol26.iloc[0:14,:].copy() # the copy function is necessary, or it throws warnings about modifying the source dataframe
exampledf.columns[:50]
exampledf.loc[:,'Vintage'] = 0.5                            # Arbitrarily set at midpoint of range
exampledf.loc[:,'Gender_1'] = 1                             # Male default
exampledf.loc[:,'Driving_License_1'] = 1                    # Has driving license
exampledf.loc[:,'Vehicle_Damage_Yes'] = 1                   # Has damage
exampledf.loc[:,'Previously_Insured_1'] = 0                 # Not previously insured
exampledf.loc[:,'Vehicle_Age_< 1 Year'] = 0                 # 
exampledf.loc[:,'Vehicle_Age_> 2 Years'] = 0                # The combination of 2 '0's here means the car is between 1 and 2 years old.  get_dummies drops that dummy variable
exampledf.loc[:,'Age_4'] = 0
exampledf.loc[:,'Age_5'] = 0
exampledf.loc[:,'Age_6'] = 0
exampledf.loc[:,'Age_7'] = 0
exampledf.loc[:,'Age_8'] = 0
exampledf.loc[:,'Age_9'] = 0    # Age_10 does not exist because it would have been first in alphabetical order, and was hence omitted.
                                # That is how get_dummies works, unless you specify you want to keep all variables.  It does not make
                                # sense for binary kinds of data like gender or driving license
exampledf.loc[:,'Age_11'] = 0
exampledf.loc[:,'Age_12'] = 0
exampledf.loc[:,'Age_13'] = 0
exampledf.loc[:,'Age_14'] = 0
exampledf.loc[:,'Age_15'] = 0
exampledf.loc[:,'Age_16'] = 0
exampledf.loc[:,'Age_17'] = 0
exampledf.loc[:,'Annual_Premium_10.0'] = 0
exampledf.loc[:,'Annual_Premium_11.0'] = 0
exampledf.loc[:,'Annual_Premium_12.0'] = 0
exampledf.loc[:,'Annual_Premium_13.0'] = 0
exampledf.loc[:,'Annual_Premium_14.0'] = 0
exampledf.loc[:,'Annual_Premium_15.0'] = 0
exampledf.loc[:,'Annual_Premium_16.0'] = 0
exampledf.loc[:,'Annual_Premium_17.0'] = 0
exampledf.loc[:,'Annual_Premium_18.0'] = 0
exampledf.loc[:,'Annual_Premium_19.0'] = 0
exampledf.loc[:,'Annual_Premium_2.0'] = 0
exampledf.loc[:,'Annual_Premium_20.0'] = 0
exampledf.loc[:,'Annual_Premium_21.0'] = 0
exampledf.loc[:,'Annual_Premium_3.0'] = 0
exampledf.loc[:,'Annual_Premium_4.0'] = 0
exampledf.loc[:,'Annual_Premium_5.0'] = 0
exampledf.loc[:,'Annual_Premium_6.0'] = 1  # This will be the default annual premium for everyone for consistency
exampledf.loc[:,'Annual_Premium_7.0'] = 0
exampledf.loc[:,'Annual_Premium_8.0'] = 0
exampledf.loc[:,'Annual_Premium_9.0'] = 0
exampledf.reset_index(drop=True, inplace=True)
#exampledf

exampledf.loc[0,'Age_4'] = 1
exampledf.loc[1,'Age_5'] = 1
exampledf.loc[2,'Age_6'] = 1
exampledf.loc[3,'Age_7'] = 1
exampledf.loc[4,'Age_8'] = 1
exampledf.loc[5,'Age_9'] = 1
exampledf.loc[7,'Age_11'] = 1
exampledf.loc[8,'Age_12'] = 1
exampledf.loc[9,'Age_13'] = 1
exampledf.loc[10,'Age_14'] = 1
exampledf.loc[11,'Age_15'] = 1
exampledf.loc[12,'Age_16'] = 1
exampledf.loc[13,'Age_17'] = 1
exampledf.iloc[:17,:18]
agedist = model.predict(exampledf)
agedist # male
import matplotlib.pyplot as plt

plt.plot(agedist[:,0])
plt.xlabel('Age bracket')
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], labels=['20','25','30','35','40','45','50','55','60','65','70','75','80','85'])
plt.ylabel('Probability')
plt.show()
model.predict(exampledf)
exampledf.loc[:,'Gender_1'] = 0
exampledf.loc[:,'Driving_License_1'] = 0
exampledf.loc[:,'Vehicle_Damage_Yes'] = 1
exampledf.loc[:,'Previously_Insured_1'] = 0
exampledf.loc[:,'Vehicle_Age_< 1 Year'] = 0
exampledf.loc[:,'Vehicle_Age_> 2 Years'] = 0
agedistf = model.predict(exampledf)
plt.plot(agedist[:,0])
plt.plot(agedistf[:,0])
plt.xlabel('Age bracket')
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], labels=['20','25','30','35','40','45','50','55','60','65','70','75','80','85'])
plt.ylabel('Probability')
plt.legend(['Male','Female'])
plt.show()
# No Driver License
exampledf.loc[:,'Gender_1'] = 1
exampledf.loc[:,'Driving_License_1'] = 0
exampledf.loc[:,'Vehicle_Damage_Yes'] = 1
exampledf.loc[:,'Previously_Insured_1'] = 0
exampledf.loc[:,'Vehicle_Age_< 1 Year'] = 0
exampledf.loc[:,'Vehicle_Age_> 2 Years'] = 0

agedistnoDL = model.predict(exampledf)
plt.plot(agedist[:,0])
plt.plot(agedistnoDL[:,0])
plt.xlabel('Age bracket')
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], labels=['20','25','30','35','40','45','50','55','60','65','70','75','80','85'])
plt.ylabel('Probability')
plt.legend(['License','No License'])
plt.show()
#No Vehicle Damage
exampledf.loc[:,'Gender_1'] = 1
exampledf.loc[:,'Driving_License_1'] = 1
exampledf.loc[:,'Vehicle_Damage_Yes'] = 0
exampledf.loc[:,'Previously_Insured_1'] = 0
exampledf.loc[:,'Vehicle_Age_< 1 Year'] = 0
exampledf.loc[:,'Vehicle_Age_> 2 Years'] = 0

agedistb = model.predict(exampledf)
plt.plot(agedist[:,0])
plt.plot(agedistb[:,0])
plt.xlabel('Age bracket')
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], labels=['20','25','30','35','40','45','50','55','60','65','70','75','80','85'])
plt.ylabel('Probability')
plt.legend(['Damage','No Damage'])
plt.show()
#Previously Insured
exampledf.loc[:,'Gender_1'] = 1
exampledf.loc[:,'Driving_License_1'] = 1
exampledf.loc[:,'Vehicle_Damage_Yes'] = 1
exampledf.loc[:,'Previously_Insured_1'] = 1
exampledf.loc[:,'Vehicle_Age_< 1 Year'] = 0
exampledf.loc[:,'Vehicle_Age_> 2 Years'] = 0

agedistb = model.predict(exampledf)
plt.plot(agedist[:,0])
plt.plot(agedistb[:,0])
plt.xlabel('Age bracket')
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], labels=['20','25','30','35','40','45','50','55','60','65','70','75','80','85'])
plt.ylabel('Probability')
plt.legend(['Not Previously Insured','Previously Insured'])
plt.show()
# Different ages of cars
exampledf.loc[:,'Gender_1'] = 1
exampledf.loc[:,'Driving_License_1'] = 1
exampledf.loc[:,'Vehicle_Damage_Yes'] = 1
exampledf.loc[:,'Previously_Insured_1'] = 0
exampledf.loc[:,'Vehicle_Age_< 1 Year'] = 1
exampledf.loc[:,'Vehicle_Age_> 2 Years'] = 0

agedistb = model.predict(exampledf)
exampledf.loc[:,'Vehicle_Age_< 1 Year'] = 0
exampledf.loc[:,'Vehicle_Age_> 2 Years'] = 1
agedistc = model.predict(exampledf)
plt.plot(agedist[:,0])
plt.plot(agedistb[:,0])
plt.plot(agedistc[:,0])
plt.xlabel('Age bracket')
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], labels=['20','25','30','35','40','45','50','55','60','65','70','75','80','85'])
plt.ylabel('Probability')
plt.legend(['1-2 years old','<1 year','>2 years'])
plt.show()
# Vintage
exampledf.loc[:,'Vintage'] = 0.25
exampledf.loc[:,'Gender_1'] = 1
exampledf.loc[:,'Driving_License_1'] = 1
exampledf.loc[:,'Vehicle_Damage_Yes'] = 1
exampledf.loc[:,'Previously_Insured_1'] = 0
exampledf.loc[:,'Vehicle_Age_< 1 Year'] = 0
exampledf.loc[:,'Vehicle_Age_> 2 Years'] = 0

agedistb = model.predict(exampledf)
exampledf.loc[:,'Vintage'] = 0.75
agedistc = model.predict(exampledf)
plt.plot(agedist[:,0])
plt.plot(agedistb[:,0])
plt.plot(agedistc[:,0])
plt.xlabel('Age bracket')
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], labels=['20','25','30','35','40','45','50','55','60','65','70','75','80','85'])
plt.ylabel('Probability')
plt.legend(['0.5','0.25','0.75'])
plt.show()
# Adjusting Annual Premium
exampledf.loc[:,'Annual_Premium_6.0'] = 0
exampledf.loc[:,'Annual_Premium_4.0'] = 1
exampledf.loc[:,'Gender_1'] = 1
exampledf.loc[:,'Driving_License_1'] = 1
exampledf.loc[:,'Vehicle_Damage_Yes'] = 1
exampledf.loc[:,'Previously_Insured_1'] = 0
exampledf.loc[:,'Vehicle_Age_< 1 Year'] = 0
exampledf.loc[:,'Vehicle_Age_> 2 Years'] = 0

agedistb = model.predict(exampledf)
exampledf.loc[:,'Annual_Premium_4.0'] = 0
exampledf.loc[:,'Annual_Premium_9.0'] = 1
agedistc = model.predict(exampledf)
plt.plot(agedist[:,0])
plt.plot(agedistb[:,0])
plt.plot(agedistc[:,0])
plt.xlabel('Age bracket')
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], labels=['20','25','30','35','40','45','50','55','60','65','70','75','80','85'])
plt.ylabel('Probability')
plt.legend(['Premium bracket 6','Premium bracket 4','Premium bracket 9'])
plt.show()
