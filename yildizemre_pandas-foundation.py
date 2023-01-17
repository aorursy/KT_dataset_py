# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon/pokemon.csv')

data.head()
# data frames from dictionary

# dictionary yazalım

ulke = ['Turkiye','Azerbeycan']

population = ['11','12']

lst_label = ["ulke",'population']

list_col = [ulke,population]

zipped = list(zip(lst_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
#add new column / yeni bir sutun

df['sehirler'] = ["istanbul",'bakü']

df
#broadcasting yeni bir column ve degerlerine 0 ata

df ['income'] = 0

df
data1 = data.loc[: ,['Attack','Defense','Speed']]

data1.plot()
data1.plot(subplots = True)

plt.show()
data1.plot(kind = "scatter",x="Attack",y = "Defense")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True)
time_list = ["1992-03-08","1992-02-25"]

print(type(time_list[1]))



datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2['date'] = datetime_object

data2 = data2.set_index('date')

data2
print(data2.loc["1992-03-10	"])

print(data2.loc["1992-03-10	":"1993-03-16	"])