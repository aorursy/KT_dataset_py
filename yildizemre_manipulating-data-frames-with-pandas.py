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
data = pd.read_csv("../input/pokemon/pokemon.csv")

data.head()
data = data.set_index("#")

data.head()
data["HP"][1]
data.HP[1]
data.loc[1,]["HP"]
data[["HP","Attack"]]
print(type(data["HP"])) #SERİES

print(type(data[["HP"]])) #DATA FRAMES
data.loc[1:10,"HP":"Defense"]
data.loc[1:10,"Speed":]
boolean = data.HP>200

data[boolean]
filtre1 = data.HP>200

filtre2 = data.Speed >20

data[filtre1 & filtre2]
def div(n):

    return n/2

data.HP.apply(div)
data.HP.apply(lambda n: n/2)
data["toplamguc"]= data.Attack + data.Defense

data.head()
print(data.index.name)

data.index.name="index_name"

data.head()
data.head()

data3= data.copy()



data3.index= range(100,900,1)

data3.head()
data1=data.set_index(["Type 1","Type 2"])

data1.head(20)