# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')

data.head
data.plot(subplots = True)

plt.show()



#alt grafiklere ayırma
data1 = data.loc[:,["Attack","Defense","Speed"]]

data1.plot()



# data plotting
data1.plot(kind = "scatter",x="Attack",y = "Defense")

plt.show()



#scatter plot olarak çizdirme
data1.plot(kind = "hist",y = "Attack",bins = 50,range= (0,250),normed = True)
data.describe()



#veriyi featureların istatistikleri ile tanımlama
fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt



time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) 

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
import warnings

warnings.filterwarnings("ignore")



data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object



data2= data2.set_index("date")

data2 



#her pokemona ait string ifadelerden date kısmında time serisi oluşturma
print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()

#sampling için daha önce oluşturduğumuz data2'yi kullanıyoruz.
data2.resample("M").mean()



# frekans sıklığı 'M'=Month olarak belirlenip resampling yapıldı.
data2.resample("M").first().interpolate("linear") 

#interpoletion ile nan olan değerlerin olduğu hücrelere lineer artacak şekilde değerler atandı.
data2.resample("M").mean().interpolate("linear")

#ortalama ile interpolate etme