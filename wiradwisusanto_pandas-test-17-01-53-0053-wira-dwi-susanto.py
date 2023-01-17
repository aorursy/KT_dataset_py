# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.r\ead_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

## WIRA DWI SUSANTO
## NIM: 17.01.53.0053

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
obj = pd.Series([1, 2, 3])
obj

## BATAS

type(obj)

## BATAS

obj2 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
obj2

## BATAS

obj2.index

## BATAS

obj2.values

## BATAS

obj2['a']

## BATAS

obj2['a'] = 4
obj2

## BATAS

obj2[['a', 'c']]

## BATAS

obj3 = pd.Series([4, 5, 6], index=['a', 'd', 'c'])
obj3

## BATAS

obj2 + obj3

## BATAS

data = {'kota': ['semarang', 'semarang', 'semarang', 'bandung', 'bandung', 'bandung'],
        'tahun': [2016, 2017, 2018, 2016, 2017, 2018],
        'populasi': [1.5, 2.1, 3.2, 2.3, 3.2, 4.5]}
frame = pd.DataFrame(data)
frame

## BATAS

type(frame)

## BATAS

frame.shape

## BATAS

frame.info()

## BATAS

frame.head()

## BATAS

frame.tail()

## BATAS

frame.columns

## BATAS

frame.index

## BATAS

frame.values

## BATAS

frame.describe()

## BATAS

frame['kota'].value_counts()

## BATAS

frame['populasi']

## BATAS

frame.loc[2]

## BATAS

frame.loc[2:3]

## BATAS

frame['populasi'][2]

## END, LANJUT KE MATPOTLIB
##

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
