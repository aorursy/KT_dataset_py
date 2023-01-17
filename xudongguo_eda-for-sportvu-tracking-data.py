# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle as pk

import json as js

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
with open('/kaggle/input/sportvu-tacking-data/0021500490.json', 'r') as f1:

    data=js.load(f1)

    df=pd.DataFrame(data)

    events=data['events']

    

#show the dataframe

print(df)

for index, value in events[0].items():

    print(index, type(value))


hometeam=events[0]['home']

visitorteam=events[0]['visitor']



for index, value in hometeam.items():

    print(index, type(value))



print('-'*130)

print('home')

print(hometeam)

print('-'*130)

print('visitor')

print(visitorteam)

hplayers=hometeam['players']

print('number of home players= ',len(hplayers))

aplayers=visitorteam['players']

print('number of away players= ',len(aplayers))

print(hplayers[0])
moments=events[0]['moments']

print('number of moment in this moments: ',len(moments))

print('-'*100)

print('moment 1:')

print(moments[0])



print('moment 1 general info:')

print(moments[0][0:5])



print('ball info:')

print(moments[0][5][0])



print('player info:')

print(moments[0][5][1])