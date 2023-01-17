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
import pandas as pd

import numpy as np



# https://www.kaggle.com/c/titanic/data より

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")



np.random.seed(666)
train.info()

print("-------------------------------------------")

print(train.isnull().sum())

print("-------------------------------------------")

print(test.isnull().sum())

print("-------------------------------------------")

train.head()
del train['PassengerId']

del train['Name']

del train['Age']

del train['Ticket']

del train['Cabin']

del train['Fare']#



del test['PassengerId']

del test['Name']

del test['Age']

del test['Ticket']

del test['Cabin']

del test['Fare']#



# 0, 1に変換

train.Sex = train.Sex.replace(['male', 'female'], [0, 1])

train.Embarked = train.Embarked.fillna('S')

train.Embarked = train.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])



test.Sex = test.Sex.replace(['male', 'female'], [0, 1])

test.Embarked = test.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])

train.head()
import keras

from keras.utils.np_utils import to_categorical

# 説明変数と目的変数に分割

y_train = train["Survived"].values



COLUMNS = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

x_train = train[COLUMNS].values

 

x_test = test[COLUMNS].values
import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Dropout



# モデル作成 正直適当

model = Sequential()

model.add(Dense(64, input_shape=(len(COLUMNS),), activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(16, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='mean_squared_error',

              optimizer='Adadelta',

              metrics=['accuracy'])
model.fit(

    x_train, 

    y_train, 

    epochs=40,

    batch_size=8, 

    verbose=2)
# test

submission = pd.read_csv("../input/titanic/gender_submission.csv")

y_test = submission["Survived"].values



[loss, accuracy] = model.evaluate(x_test, y_test)

print("loss:{0} -- accuracy:{1}".format(loss, accuracy))
predictions = model.predict(x_test)



# テスト値を再読み込みして，SVMでクラス分類したカラムを追加

df_out = pd.read_csv("../input/titanic/test.csv")

# numpyのroundが銀行丸めのため意図した結果に傾かなったので普通の四捨五入を採用

#df_out["Survived"] = np.ceil(np.round(predictions, decimals=1)).astype(np.int)

df_out["Survived"] = list(map(lambda x:int((x*2+1)/2), np.round(predictions, decimals=1)))

print(df_out[["PassengerId","Survived"]])



# outputディレクトリに出力する

df_out[["PassengerId","Survived"]].to_csv("submission.csv",index=False)