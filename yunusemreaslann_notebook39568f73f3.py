# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import datetime



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv')



data.head()

data.tail()
data.info()# for the information



"""

TimeStamp is not  method ı want to . I need the convert TimeStamp to years

for each in data.Timestamp:

    each = datetime.datetime.fromtimestamp(each).year

data.Timestamp  = [ datetime.datetime.fromtimestamp(each).year for each in data.Timestamp ]

data.plot(kind = 'scatter',x = 'Timestamp', y = 'Weighted_Price' , alpha = 0.3 ,color = 'red',grid = True )

plt.xlabel('Timestamp')

plt.ylabel('Weighted_Price' )

plt.title('BTC_change_over_time')

plt.show()

"""

data.Timestamp  = [ datetime.datetime.fromtimestamp(each).year for each in data.Timestamp ]

data.plot(kind = 'scatter',x = 'Timestamp', y = 'Weighted_Price' ,label = 'Scatter' ,alpha = 0.9 ,color = 'gray',grid = True,edgecolor= 'yellow' )

plt.xlabel('Timestamp')

plt.ylabel('Weighted_Price' )

plt.title('BTC_change_over_time')

plt.legend(loc = 'lower right')

plt.show()





data.plot(kind = 'line', x = 'Open',y = 'Close',color = 'red',linewidth = 0.2 ,label = 'Open and Close',linestyle = '--')

plt.xlabel('Open'  )

plt.ylabel('Close' )

plt.legend(loc = 'lower right')

plt.title('Line of Open and Close')

plt.show()
data.Timestamp.plot(kind = 'hist',color = 'red',bins = 100 , figsize= (10,10))

plt.xlabel('Timestamp')

plt.ylabel('frekans' )

plt.title('hist')

plt.legend()

plt.show()
dict1 = {"NAME":["N1","N2","N3","N4"],

          "Surname":["S1","S2","S3","S4"]}

# dict creation



print(dict1.keys())

print(dict1.values())

dict1["NAME"]  = "N5"  # changes all values 

print(dict1.values())

dict1

dict2 = {"AGE" :  [12 , 34 , 43 ,23]} 

dict1.update(dict2)

print(dict1)

dict2.clear() #remove entries

print(dict2)

dict1_pd = pd.DataFrame(dict1) # conver to data frame

dict1['AGE'] = [10,12,14,13]



dict1_pd.head()   # show with pandas 











series = data['Weighted_Price']        

print(type(series))

data_frame = data[['Weighted_Price']]  

print(type(data_frame))    # different clases creation
# type = boolen ( True or False)



print(1>2) # this is  false

print(2>1) # this is true

# we have accepted logic materials for boolean data types

print(True and False) # &

print(True or False)


data[(data['Weighted_Price']>4000) & (5000 >data['Weighted_Price'])]
data[np.logical_and(data['Weighted_Price']>5432 ,  5600 > data['High'])]
liste  =  [ 1,2,3,4,5 ]



for i in liste: 

    print( i -1 , ":" , i )

    



for index, value in enumerate(liste):

    print(index , ": ", value)

    

    
dictionary = {"samsun" :" ilkadim","istanbul":"kadikoy","ordu":"unye"}

for  key , value in dictionary.items():

    print(key , ":" , value)

    

for index ,value in data[["Timestamp"]][:1].iterrows():

    print(index , ":" , value)
i = 4

while i < 5:

    print(i)

    i += 1

print(i)





    