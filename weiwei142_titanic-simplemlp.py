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
from sklearn import preprocessing

#np.random.seed(10)
all_df = pd.read_csv('../input/train.csv')
all_df[:10]
col = ['Survived','Name','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

all_df = all_df[col]
Gender_df = all_df[['Survived','Sex']]

Gender_num=Gender_df.groupby("Sex").count()

Gender_num
Gender_Survived = Gender_df.groupby("Survived").count()

Gender_Survived
Gender_male = Gender_df.groupby("Sex").get_group("male")

Gender_male_survived = Gender_male[Gender_male['Survived']==1].shape[0]

Gender_female = Gender_df.groupby("Sex").get_group("female")

Gender_female_survived = Gender_female[Gender_female['Survived']==1].shape[0]

Gender_Survived_df = pd.DataFrame({'Male':[Gender_male_survived],'Female':[Gender_female_survived]})

Gender_Survived
import seaborn as sns

sns.countplot(all_df['Sex'],hue=all_df["Survived"])

all_df[['Sex','Survived']].groupby('Sex').mean().round(3)
sns.countplot(all_df['Pclass'],hue=all_df['Survived'])

all_df[['Pclass','Survived']].groupby('Pclass').mean().round(3)
def PreprocessData(all_df):

    df = all_df.drop(['Name'],axis=1) #名字不影響存活率

    age_mean = df['Age'].mean() 

    df['Age'] = df['Age'].fillna(age_mean) #年紀當中有NaNull值（不詳），將平均值帶入

    fare_mean = df['Fare'].mean()

    df['Fare'] = df['Fare'].fillna(fare_mean) #票價同理

    df['Sex'] = df['Sex'].map({'female':0,'male':1}).astype(int)

    X_OneHot_df = pd.get_dummies(data=df,columns=['Embarked']) #值之間沒有大小意義，可以使用One-Hot

    print(X_OneHot_df)

    ndarray = X_OneHot_df.values

    Features = ndarray[:,1:]

    Label = ndarray[:,0]

    

    #將數據正規化（Range : 0~1）

    minmax_sacle = preprocessing.MinMaxScaler(feature_range=(0,1))

    scaledFeatures = minmax_sacle.fit_transform(Features)

    #

    return scaledFeatures, Label

PreprocessData(all_df)

    
np.random.seed(10)

msk = np.random.rand(len(all_df)) < 0.8

train_df = all_df[msk]

test_df = all_df[~msk]

train_Features, train_Label = PreprocessData(train_df)

test_Features, test_Label = PreprocessData(test_df)
from keras.models import Sequential

from keras.layers import Dense,Dropout

model = Sequential()

model.add(Dense(units=40,input_dim=9,kernel_initializer='uniform',activation='relu'))

model.add(Dense(units=30,kernel_initializer='uniform',activation='relu'))

model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

train_history = model.fit(x=train_Features,y=train_Label,validation_split=0.1,epochs=50,batch_size=30,verbose=2)
import matplotlib.pyplot as plt

def show_train_history(train_history, train, validation):

    plt.plot(train_history.history[train])

    plt.plot(train_history.history[validation])

    plt.title('Trian History')

    plt.ylabel(train)

    plt.xlabel("Epochs")

    plt.legend(['train','validation'],loc='lower right')

    plt.show()

show_train_history(train_history,'acc','val_acc')

show_train_history(train_history,'loss','val_loss')
scores = model.evaluate(x=test_Features,y=test_Label)

pd.DataFrame(scores,index=model.metrics_names,columns=[''])
all_f, all_label = PreprocessData(all_df)

all_probability = model.predict(all_f)

all_df['Probability']=all_probability

all_df
predict_data = pd.read_csv('../input/test.csv')

predict_data = predict_data.sort_values(by = 'PassengerId')

col = ['Name','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'] #並沒有Survived這個欄位（未知）

predict_data = predict_data[col]

# 由於並沒有Survived這個欄位，PreprocessData函式已不可用

def predict_data_preprocess(all_df):

    df = all_df.drop(['Name'],axis=1) #名字不影響存活率

    age_mean = df['Age'].mean() 

    df['Age'] = df['Age'].fillna(age_mean) #年紀當中有NaNull值（不詳），將平均值帶入

    fare_mean = df['Fare'].mean()

    df['Fare'] = df['Fare'].fillna(fare_mean) #票價同理

    df['Sex'] = df['Sex'].map({'female':0,'male':1}).astype(int)

    X_OneHot_df = pd.get_dummies(data=df,columns=['Embarked']) #值之間沒有大小意義，可以使用One-Hot

    print(X_OneHot_df)

    ndarray = X_OneHot_df.values

    Features = ndarray

    

    #將數據正規化（Range : 0~1）

    minmax_sacle = preprocessing.MinMaxScaler(feature_range=(0,1))

    scaledFeatures = minmax_sacle.fit_transform(Features)

    #

    return scaledFeatures

predict_data = predict_data_preprocess(predict_data)
test = model.predict(predict_data)

test
fliter = np.around(test).astype(int)

submit = pd.read_csv('../input/gender_submission.csv')

submit['Survived'] = fliter

submit
submit.to_csv('./submit.csv', index=False)