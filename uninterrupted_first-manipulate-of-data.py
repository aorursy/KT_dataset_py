# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
data.head()
data.columns
data.Category

    
#kategori içindeki aynı isimleri silerek, farklı kategorileri göstermek için böyle uzun ve anlamsız bir kod yazıldı...

l=[]

i=0

for cat in data.Category:



    

    if i>0:

        if j == cat:

            i=1

        else:

            if (cat in l):

                i=1

            else:

                l.append(cat)

                

    i=1  

    j=cat

    

print(l)    
x = data['Rating'] > 4.6

data[x]
y = data['Category'] == 'EDUCATION'

data[y]

data[(data['Rating']>4.6) & (data['Category']=='EDUCATION')]#Eğitim alanındaki 4.6 üzeri rating alan appler

data[(data['Rating']>4.6) & (data['Category']=='EDUCATION') & (data['Type']=='Free')]   #üsttekinin bedava olanları

#indirlme oranı ile rating arasındaki ilişki



plot.scatter(data.Rating[0:300],data.Installs[0:300])

plot.xlabel('Ratings')

plot.ylabel('Installs')

plot.show()



data.Rating[0:300].plot(kind='hist', bins=10, figsize=(13,13))