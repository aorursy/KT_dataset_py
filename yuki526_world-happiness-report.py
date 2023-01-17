# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model

from keras.layers import Activation, Dense, Dropout

from keras import optimizers

from keras.utils.np_utils import to_categorical

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#2015、2016年データで2017年のランクを予想

data2015 = pd.read_csv('../input/world-happiness/2015.csv')

data2016 = pd.read_csv('../input/world-happiness/2016.csv')
data2015.columns
data2016.columns
#2015年のカラムうち2016年にないものを削除

data2015 = data2015.drop('Standard Error', axis=1)

data2015 = data2015.drop(132)
#2016年のカラムうち2015年にないものを削除

data2016 = data2016.drop('Lower Confidence Interval', axis=1)

data2016 = data2016.drop('Upper Confidence Interval', axis=1)
#2015年・2016年のデータを縦に連結

newdata = pd.concat([data2015, data2016], axis=0) 

newdata.isnull().sum()

#newdata.to_csv("newdata.csv", index=False)
# 目的変数

y = newdata['Happiness Score']



#説明変数(Happiness Rankは結果に直結しているので除外)

allCols = ['Economy (GDP per Capita)', 'Family','Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)','Generosity', 'Dystopia Residual']

x = newdata[allCols]



#x_train = x

#y_train = y



#学習用、検証用データに分割

(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size = 0.1, random_state=0,)
model = Sequential()



model.add(Dense(64, input_dim=7))

model.add(Activation("relu"))



#model.add(Dense(16))

#model.add(Activation("relu"))



#model.add(Dense(16))

#model.add(Activation("sigmoid"))

#model.add(Dropout(rate=0.5))



model.add(Dense(1))
#sgd = optimizers.SGD(lr=0.1)

model.compile(loss="mean_squared_error", optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=16, verbose=1,validation_data=(x_test, y_test))
test_data = y_test

test_label = model.predict(x_test)
#np.arrayをDFに変更、y_test(test_data)にindex番号を振りなおして”index”列を削除する

test_data = pd.DataFrame(test_data).reset_index().drop('index', axis=1)

test_label = pd.DataFrame(test_label)

test_label.columns = ['Happiness Score']
result = pd.concat([test_data, test_label] ,axis=1)

result.columns = ["acc score", "pred score"]

result.head(15)
#2017年のデータ読み込み、いらないカラム削除

data2017 = pd.read_csv('../input/world-happiness/2017.csv')

data2017 = data2017.drop('Whisker.high', axis=1)

data2017 = data2017.drop('Whisker.low', axis=1)
#2017年のカラム名を対応しているものに変更

data2017 = data2017.rename(columns={'Health..Life.Expectancy.': 'Health (Life Expectancy)', 'Trust..Government.Corruption.': 'Trust (Government Corruption)', 'Economy..GDP.per.Capita.': 'Economy (GDP per Capita)', 'Dystopia.Residual': 'Dystopia Residual'})
pred_y = data2017['Happiness.Score']



#説明変数(Happiness Rankが直結しているので除外)

allCols = ['Economy (GDP per Capita)', 'Family','Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)','Generosity', 'Dystopia Residual']

pred_x = data2017[allCols]
prediction = model.predict(pred_x)

prediction = pd.DataFrame(prediction)

prediction.columns = ['prHappiness Score']

final_predict = pd.concat([prediction, data2017['Country']], axis=1)

final_predict = final_predict.sort_values('prHappiness Score', ascending=False)

final_predict = pd.DataFrame(final_predict)
pred_y = pd.DataFrame(pred_y)

finaldata2017 = pd.concat([data2017['Happiness.Score'], data2017['Country']], axis=1)

finaldata2017  = pd.DataFrame(finaldata2017)
results = pd.concat([finaldata2017, final_predict], axis=1)

results.head(50)
data2018 = pd.read_csv('../input/whr2018/WorldHappiness2018_Data.csv')
data2018.columns
data2017.columns

#対応しているものが多いが、Social_Suppport = Faimly？？

#Social support = 周囲の人々から与えられる物質的・心理的支援とのことなのでfamilyと同意義ととらえる。
#2018年のカラム名を対応しているものに変更

data2018 = data2018.rename(columns={'Healthy_Life_Expectancy': 'Health (Life Expectancy)', 'Perceptions_Of_Corruption': 'Trust (Government Corruption)',  'GDP_Per_Capita': 'Economy (GDP per Capita)', 'Residual': 'Dystopia Residual', 'Freedom_To_Make_Life_Choices': 'Freedom', 'Social_Support': 'Family'})

data2018
pred_y2018 = data2018['Score']



#説明変数(Happiness Rankが直結しているので除外)

allCols = ['Economy (GDP per Capita)', 'Family','Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)','Generosity', 'Dystopia Residual']

pred_x2018 = data2018[allCols]
prediction2018 = model.predict(pred_x2018)

prediction2018 = pd.DataFrame(prediction2018)

prediction2018.columns = ['prHappiness Score']

final_predict2018 = pd.concat([prediction2018, data2018['Country']], axis=1)

final_predict2018 = final_predict2018.sort_values('prHappiness Score', ascending=False)

final_predict2018 = pd.DataFrame(final_predict2018)



pred_y2018 = pd.DataFrame(pred_y2018)

finaldata2018 = pd.concat([data2018['Score'], data2018['Country']], axis=1)

finaldata2018  = pd.DataFrame(finaldata2018)



results2018 = pd.concat([finaldata2018, final_predict2018], axis=1)

results2018.head(50)
import pandas as pd

data2019 = pd.read_csv('../input/whr2019/world-happiness-report-2019.csv')

data2019.isnull().sum()
#欠損値が多い変数を埋める（平均値で埋めてみる）Corruption, GDP, Healthy

data2019[['Corruption']] = data2019[['Corruption']].fillna(value=data2019[['Corruption']].mean())

data2019[['Log of GDP\nper capita']] = data2019[['Log of GDP\nper capita']].fillna(value=data2019[['Log of GDP\nper capita']].mean())

data2019[['Healthy life\nexpectancy']] = data2019[['Healthy life\nexpectancy']].fillna(value=data2019[['Healthy life\nexpectancy']].mean())
#正規化（正しくできてるかは不明）

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



dfArray = data2019.values

dfArray = np.delete(dfArray, 0, axis=1)

dfArray = np.delete(dfArray, 1, axis=1)

ms = MinMaxScaler()

dfArray = ms.fit_transform(dfArray)

dfArray = pd.DataFrame(dfArray)

dfArray = dfArray.sort_values(0, ascending=False)

dfArray = dfArray.reset_index().drop('index', axis=1)

dfArray
column_names = ['Country (region)', 'Ladder', 'SD of Ladder', 'Positive affect',

       'Negative affect', 'Social support', 'Freedom', 'Corruption',

       'Generosity', 'Log of GDP\nper capita', 'Healthy life\nexpectancy']

new_data2019 = pd.concat([data2019['Country (region)'], data2019['Ladder'], dfArray], axis=1,)

new_data2019.columns = column_names

new_data2019
#2019年のカラム名を対応しているものに変更

new_data2019 = new_data2019.rename(columns={'Healthy life\nexpectancy': 'Health (Life Expectancy)', 'Corruption': 'Trust (Government Corruption)',  'Log of GDP\nper capita': 'Economy (GDP per Capita)', 'Social support': 'Family'})

new_data2019['Dystopia Residual'] = 0
#説明変数(Happiness Rankが直結しているので除外)

allCols = ['Economy (GDP per Capita)', 'Family','Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)','Generosity','Dystopia Residual']

pred_x2019 = new_data2019[allCols]



prediction2019 = model.predict(pred_x2019)

prediction2019 = pd.DataFrame(prediction2019)

prediction2019.columns = ['prHappiness Score']

pred2019 = pd.concat([data2019['Country (region)'], prediction2019], axis=1,)

pred2019 = pred2019.sort_values('prHappiness Score', ascending=False )

pred2019