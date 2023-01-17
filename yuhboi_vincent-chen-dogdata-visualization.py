# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Dogs_of_Cambridge = pd.read_csv("/kaggle/input/vincent-dog-sub/clean_dogs_Vincent_Chen.csv")

Dogs_of_Cambridge.describe
Dogs_of_Cambridge.plot(x ='Latitude_masked', y='Longitude_masked', kind = 'scatter')
Neighborhoods = {}

for neigh in Dogs_of_Cambridge.Neighborhood.unique():

    Neighborhoods.update({neigh:0})

print (Neighborhoods)

for i in range(len(Dogs_of_Cambridge.Dog_Name)):

    Neighborhoods[Dogs_of_Cambridge.Neighborhood[i]] += 1

key = []

val = []

for i in Neighborhoods.keys():

    key.append(i)

    val.append(Neighborhoods[i])

    

df = pd.DataFrame({'freq': val}, index=key)

plot = df.plot.pie(y='freq', figsize=(5, 5))

print(plot)
df = pd.DataFrame({'freq': val,'neighborhoods': key},)

plot = df.plot(x ='neighborhoods', y='freq', kind = 'bar')

print(plot)
def allBreed():

    arr = {}

    for i in range(len(Dogs_of_Cambridge.Dog_Name)):

        if Dogs_of_Cambridge.Dog_Breed[i] not in arr:

            arr.update({Dogs_of_Cambridge.Dog_Breed[i]:1})

        else:

            arr[Dogs_of_Cambridge.Dog_Breed[i]] += 1

    return arr

breeds = []

frequencies = []

for i in allBreed().keys():

    breeds.append(i)

    frequencies.append(allBreed()[i])

    print(i)

Data = {'Breed': breeds, 'Frequency': frequencies}

  

bar = DataFrame(Data,columns=['Breed','Frequency'])

print (bar)
plot = bar.plot(x ='Breed', y='Frequency', kind = 'bar')

print(plot)
df = pd.DataFrame({'Frequency': val}, index=breeds)

plot = bar.plot.pie(y='Frequency', figsize=(5, 5))

print(plot)