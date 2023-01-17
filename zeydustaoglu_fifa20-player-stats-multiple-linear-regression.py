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
df20 = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')
df20.head()
#modelimizde kullanacagimiz kolonlari data setimizden cekiyoruz.
short_name = df20[['short_name']]
potential = df20[['potential']]
age = df20[['age']]
value_eur = df20[['value_eur']]
pace = df20[['pace']]
shooting = df20[['shooting']]
passing = df20[['passing']]
dribbling = df20[['dribbling']]
defending = df20[['defending']]

#kolonlari concat ile birlestirdik yeni bir dataframe olusturduk
df_new = pd.concat([short_name,potential,value_eur,pace,shooting,passing,defending], axis = 1).dropna()
df_new.head()
df_new.info()
#kolonlarin yerini degistirdik. value kolonunu sona aldik

df_new = df_new[['short_name', 'potential','pace','shooting','passing','defending','value_eur']]
df_new.head()


#oyuncunun yasini potansiyelini, sut yetenegini, pas yetenegini,dripling yetenegi ve defans yetenegini modelimize ogretip
#oyuncunun piyasa degerini tahmin ettirmeye calisacagiz

from sklearn.linear_model import LinearRegression

x = df_new.iloc[:,[1,3]].values 
y = df_new.value_eur.values.reshape(-1,1) #valueyu da y ye atadik

# %% fitting data
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0: ", multiple_linear_regression.intercept_)
print("b1,b2: ",multiple_linear_regression.coef_)

# predict
print(multiple_linear_regression.predict(np.array([[94,90]])))
